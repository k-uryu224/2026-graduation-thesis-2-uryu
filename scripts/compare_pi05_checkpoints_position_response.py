#!/usr/bin/env python3

import csv
import gc
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

try:
    from lerobot.utils.control_utils import (
        prepare_observation_for_inference,
    )
except ImportError:
    from lerobot.common.control_utils import (
        prepare_observation_for_inference,
    )


ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "results/lerobot_datasets/"
    "so101_pick_object_train_60eps"
)

CHECKPOINTS = [
    "005000",
    "010000",
    "015000",
    "020000",
    "025000",
    "030000_recovered",
]

CHECKPOINT_ROOT = (
    ROOT
    / "outputs/train/pi05_teacher_v1/checkpoints"
)

OUTPUT_DIR = (
    ROOT
    / "results/evaluation/pi05_teacher_v1/"
    "checkpoint_position_response_v1"
)

TASK = "Pick up the object."
ROBOT_TYPE = "so101_mujoco"

FRAMES_PER_EPISODE = 210

POSITIONS = [
    (0.220, 0),
    (0.240, 20),
    (0.260, 40),
]

TEST_FRAMES = [
    0,
    50,
    100,
    150,
]

NOISE_SEED = 20260920


def image_tensor_to_raw_numpy(image):
    image = image.detach().cpu()
    image = image.permute(1, 2, 0).contiguous()

    if torch.is_floating_point(image):
        image = (
            image.clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
        )
    else:
        image = image.to(torch.uint8)

    return image.numpy()


def build_batch(
    *,
    sample,
    preprocessor,
    device,
):
    observation = {
        "observation.images.top":
            image_tensor_to_raw_numpy(
                sample["observation.images.top"]
            ),
        "observation.images.wrist":
            image_tensor_to_raw_numpy(
                sample["observation.images.wrist"]
            ),
        "observation.state":
            sample["observation.state"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
    }

    batch = prepare_observation_for_inference(
        observation,
        device,
        TASK,
        ROBOT_TYPE,
    )

    return preprocessor(batch)


def to_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())

    return int(value)


def collect_experts(dataset):
    experts = {}

    for object_x, episode_index in POSITIONS:
        episode_start = (
            episode_index
            * FRAMES_PER_EPISODE
        )

        first = dataset[episode_start]

        actual_episode = to_int(
            first["episode_index"]
        )

        if actual_episode != episode_index:
            raise RuntimeError(
                f"Episode mismatch: "
                f"x={object_x:.3f}, "
                f"expected={episode_index}, "
                f"actual={actual_episode}"
            )

        experts[object_x] = {}

        for frame in TEST_FRAMES:
            global_index = (
                episode_start + frame
            )

            chunk = []

            for offset in range(50):
                sample = dataset[
                    global_index + offset
                ]

                actual_ep = to_int(
                    sample["episode_index"]
                )

                if actual_ep != episode_index:
                    raise RuntimeError(
                        "Expert chunk crossed "
                        "episode boundary."
                    )

                chunk.append(
                    sample["action"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            experts[object_x][frame] = (
                np.stack(chunk, axis=0)
            )

    return experts


def collect_predictions(
    *,
    dataset,
    policy,
    preprocessor,
    postprocessor,
    device,
):
    predictions = {}

    for object_x, episode_index in POSITIONS:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        generator = torch.Generator(
            device=device
        )
        generator.manual_seed(
            NOISE_SEED
        )

        noise_shape = (
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
        )

        episode_start = (
            episode_index
            * FRAMES_PER_EPISODE
        )

        predictions[object_x] = {}

        for frame in TEST_FRAMES:
            sample = dataset[
                episode_start + frame
            ]

            batch = build_batch(
                sample=sample,
                preprocessor=preprocessor,
                device=device,
            )

            noise = torch.randn(
                noise_shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

            with torch.inference_mode():
                raw = (
                    policy.predict_action_chunk(
                        batch,
                        noise=noise,
                    )
                )

                processed = postprocessor(
                    raw.clone()
                )

            predictions[object_x][frame] = (
                processed[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return predictions


def flatten_arm(source, object_x):
    return np.concatenate(
        [
            source[object_x][frame][:, :5]
            for frame in TEST_FRAMES
        ],
        axis=0,
    ).reshape(-1)


def arm_rmse(pred, expert):
    error = pred[:, :5] - expert[:, :5]

    return float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )


def aggregate_self_rmse(
    predictions,
    experts,
    object_x,
):
    pred = np.concatenate(
        [
            predictions[object_x][frame]
            for frame in TEST_FRAMES
        ],
        axis=0,
    )

    expert = np.concatenate(
        [
            experts[object_x][frame]
            for frame in TEST_FRAMES
        ],
        axis=0,
    )

    return arm_rmse(
        pred,
        expert,
    )


def project_segment(
    point,
    start,
    end,
    x_start,
    x_end,
):
    direction = end - start

    denom = float(
        np.dot(
            direction,
            direction,
        )
    )

    if denom <= 1e-12:
        raise RuntimeError(
            "Degenerate expert segment."
        )

    t = float(
        np.dot(
            point - start,
            direction,
        )
        / denom
    )

    t_clipped = min(
        1.0,
        max(0.0, t),
    )

    projection = (
        start
        + t_clipped * direction
    )

    residual = float(
        np.sqrt(
            np.mean(
                (point - projection) ** 2
            )
        )
    )

    effective_x = (
        x_start
        + t_clipped
        * (x_end - x_start)
    )

    return (
        effective_x,
        residual,
    )


def effective_x(
    predictions,
    experts,
    pred_x,
):
    point = flatten_arm(
        predictions,
        pred_x,
    )

    e220 = flatten_arm(
        experts,
        0.220,
    )

    e240 = flatten_arm(
        experts,
        0.240,
    )

    e260 = flatten_arm(
        experts,
        0.260,
    )

    candidates = [
        project_segment(
            point,
            e220,
            e240,
            0.220,
            0.240,
        ),
        project_segment(
            point,
            e240,
            e260,
            0.240,
            0.260,
        ),
    ]

    return min(
        candidates,
        key=lambda x: x[1],
    )


def response_ratio(
    *,
    predictions,
    experts,
    x1,
    x2,
    joint_index,
):
    pred_response = np.concatenate(
        [
            (
                predictions[x2][frame]
                - predictions[x1][frame]
            )[:, joint_index]
            for frame in TEST_FRAMES
        ]
    )

    expert_response = np.concatenate(
        [
            (
                experts[x2][frame]
                - experts[x1][frame]
            )[:, joint_index]
            for frame in TEST_FRAMES
        ]
    )

    pred_rms = float(
        np.sqrt(
            np.mean(
                pred_response ** 2
            )
        )
    )

    expert_rms = float(
        np.sqrt(
            np.mean(
                expert_response ** 2
            )
        )
    )

    return (
        pred_rms / expert_rms
        if expert_rms > 1e-12
        else float("nan")
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda")

    print("=== load dataset ===")

    dataset = LeRobotDataset(
        repo_id=(
            "local/"
            "so101_pick_object_train_60eps"
        ),
        root=DATASET_ROOT,
        video_backend="pyav",
    )

    experts = collect_experts(
        dataset
    )

    rows = []

    for checkpoint_name in CHECKPOINTS:
        checkpoint = (
            CHECKPOINT_ROOT
            / checkpoint_name
            / "pretrained_model"
        )

        print()
        print("=" * 80)
        print(
            "CHECKPOINT:",
            checkpoint_name,
        )

        policy = PI05Policy.from_pretrained(
            str(checkpoint)
        )
        policy = policy.to(device)
        policy.eval()

        preprocessor, postprocessor = (
            make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=str(
                    checkpoint
                ),
                preprocessor_overrides={
                    "device_processor": {
                        "device": "cuda",
                    },
                },
            )
        )

        predictions = collect_predictions(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )

        result = {
            "checkpoint":
                checkpoint_name,
        }

        for object_x, _ in POSITIONS:
            rmse = (
                aggregate_self_rmse(
                    predictions,
                    experts,
                    object_x,
                )
            )

            eff_x, residual = (
                effective_x(
                    predictions,
                    experts,
                    object_x,
                )
            )

            bias_mm = (
                eff_x - object_x
            ) * 1000.0

            result[
                f"arm_rmse_x{object_x:.3f}"
            ] = rmse

            result[
                f"effective_x_x{object_x:.3f}"
            ] = eff_x

            result[
                f"bias_mm_x{object_x:.3f}"
            ] = bias_mm

            result[
                f"projection_residual_x{object_x:.3f}"
            ] = residual

        for x1, x2, label in [
            (
                0.220,
                0.240,
                "220_240",
            ),
            (
                0.240,
                0.260,
                "240_260",
            ),
        ]:
            result[
                f"shoulder_lift_ratio_{label}"
            ] = response_ratio(
                predictions=predictions,
                experts=experts,
                x1=x1,
                x2=x2,
                joint_index=1,
            )

            result[
                f"elbow_flex_ratio_{label}"
            ] = response_ratio(
                predictions=predictions,
                experts=experts,
                x1=x1,
                x2=x2,
                joint_index=2,
            )

        rows.append(result)

        print(
            "self arm RMSE:",
            f".220={result['arm_rmse_x0.220']:.4f}",
            f".240={result['arm_rmse_x0.240']:.4f}",
            f".260={result['arm_rmse_x0.260']:.4f}",
        )

        print(
            "effective x:",
            f".220->{result['effective_x_x0.220']:.5f}",
            f"({result['bias_mm_x0.220']:+.1f}mm)",
            f".240->{result['effective_x_x0.240']:.5f}",
            f"({result['bias_mm_x0.240']:+.1f}mm)",
            f".260->{result['effective_x_x0.260']:.5f}",
            f"({result['bias_mm_x0.260']:+.1f}mm)",
        )

        print(
            "response ratio .220->.240:",
            f"shoulder="
            f"{result['shoulder_lift_ratio_220_240']:.3f}",
            f"elbow="
            f"{result['elbow_flex_ratio_220_240']:.3f}",
        )

        print(
            "response ratio .240->.260:",
            f"shoulder="
            f"{result['shoulder_lift_ratio_240_260']:.3f}",
            f"elbow="
            f"{result['elbow_flex_ratio_240_260']:.3f}",
        )

        del predictions
        del preprocessor
        del postprocessor
        del policy

        gc.collect()
        torch.cuda.empty_cache()

    csv_path = (
        OUTPUT_DIR
        / "checkpoint_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("=== CHECKPOINT SUMMARY ===")

    for row in rows:
        print(
            row["checkpoint"],
            "RMSE",
            f"{row['arm_rmse_x0.220']:.4f}",
            f"{row['arm_rmse_x0.240']:.4f}",
            f"{row['arm_rmse_x0.260']:.4f}",
            "| bias mm",
            f"{row['bias_mm_x0.220']:+.1f}",
            f"{row['bias_mm_x0.240']:+.1f}",
            f"{row['bias_mm_x0.260']:+.1f}",
            "| response .240->.260",
            f"{row['shoulder_lift_ratio_240_260']:.3f}",
            f"{row['elbow_flex_ratio_240_260']:.3f}",
        )

    print()
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
