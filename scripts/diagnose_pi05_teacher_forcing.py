#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

try:
    from lerobot.utils.control_utils import prepare_observation_for_inference
except ImportError:
    from lerobot.common.control_utils import prepare_observation_for_inference


ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "results/lerobot_datasets/so101_pick_object_train_60eps"
)

CHECKPOINT = (
    ROOT
    / "outputs/train/pi05_teacher_v1"
    / "checkpoints/030000_recovered/pretrained_model"
)

TASK = "Pick up the object."
ROBOT_TYPE = "so101_mujoco"

EPISODE_INDEX = 20
FRAMES_PER_EPISODE = 210
TEST_FRAMES = [0, 50, 100, 150]

NOISE_SEED = 20260920

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def image_tensor_to_raw_numpy(image):
    image = image.detach().cpu()

    image = (
        image.permute(1, 2, 0)
        .contiguous()
    )

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


def build_processed_batch(
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


def main():
    device = torch.device("cuda")

    print("=== load dataset ===")

    dataset = LeRobotDataset(
        repo_id="local/so101_pick_object_train_60eps",
        root=DATASET_ROOT,
        video_backend="pyav",
    )

    episode_start = (
        EPISODE_INDEX
        * FRAMES_PER_EPISODE
    )

    first_sample = dataset[episode_start]

    print("dataset frames:", len(dataset))
    print("episode index:", EPISODE_INDEX)
    print("global start:", episode_start)

    if "episode_index" in first_sample:
        actual_episode = to_int(
            first_sample["episode_index"]
        )
        print(
            "dataset episode_index:",
            actual_episode,
        )

        if actual_episode != EPISODE_INDEX:
            raise RuntimeError(
                "Episode mapping mismatch: "
                f"{actual_episode}"
            )

    print("\n=== load policy ===")

    policy = PI05Policy.from_pretrained(
        str(CHECKPOINT)
    )
    policy = policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(CHECKPOINT),
            preprocessor_overrides={
                "device_processor": {
                    "device": "cuda",
                },
            },
        )
    )

    preprocessor.reset()
    postprocessor.reset()

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(NOISE_SEED)

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    print()
    print("=== teacher-forcing diagnostic ===")
    print("noise seed:", NOISE_SEED)

    all_errors = []

    for chunk_index, frame in enumerate(
        TEST_FRAMES
    ):
        global_index = episode_start + frame

        sample = dataset[global_index]

        batch = build_processed_batch(
            sample=sample,
            preprocessor=preprocessor,
            device=device,
        )

        # Draw noises in exactly the same sequence
        # as closed-loop replanning.
        chunk_noise = torch.randn(
            noise_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            raw_chunk = (
                policy.predict_action_chunk(
                    batch,
                    noise=chunk_noise,
                )
            )

            pred_chunk = postprocessor(
                raw_chunk.clone()
            )

        pred = (
            pred_chunk[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        expert = []

        for offset in range(
            policy.config.chunk_size
        ):
            idx = global_index + offset

            # All tested chunks remain inside
            # episode 20.
            expert_sample = dataset[idx]

            if "episode_index" in expert_sample:
                ep = to_int(
                    expert_sample[
                        "episode_index"
                    ]
                )

                if ep != EPISODE_INDEX:
                    raise RuntimeError(
                        "Expert chunk crossed "
                        "episode boundary."
                    )

            expert.append(
                expert_sample["action"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        expert = np.stack(
            expert,
            axis=0,
        )

        error = pred - expert

        mae_joint = np.mean(
            np.abs(error),
            axis=0,
        )

        rmse_joint = np.sqrt(
            np.mean(
                error ** 2,
                axis=0,
            )
        )

        total_rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        all_errors.append(error)

        print()
        print(
            f"--- chunk {chunk_index} "
            f"frame={frame} ---"
        )

        print(
            "total raw-space RMSE:",
            f"{total_rmse:.6f}",
        )

        for i, name in enumerate(
            ACTION_NAMES
        ):
            print(
                f"{name:16s} "
                f"MAE={mae_joint[i]:.6f} "
                f"RMSE={rmse_joint[i]:.6f}"
            )

        print("expert first:", expert[0])
        print("pred   first:", pred[0])
        print("expert last :", expert[-1])
        print("pred   last :", pred[-1])

        print(
            "gripper expert min/max:",
            float(expert[:, 5].min()),
            float(expert[:, 5].max()),
        )
        print(
            "gripper pred   min/max:",
            float(pred[:, 5].min()),
            float(pred[:, 5].max()),
        )

    all_errors = np.concatenate(
        all_errors,
        axis=0,
    )

    print()
    print("=== aggregate ===")

    aggregate_mae = np.mean(
        np.abs(all_errors),
        axis=0,
    )

    aggregate_rmse = np.sqrt(
        np.mean(
            all_errors ** 2,
            axis=0,
        )
    )

    for i, name in enumerate(
        ACTION_NAMES
    ):
        print(
            f"{name:16s} "
            f"MAE={aggregate_mae[i]:.6f} "
            f"RMSE={aggregate_rmse[i]:.6f}"
        )

    print(
        "overall raw-space RMSE:",
        float(
            np.sqrt(
                np.mean(
                    all_errors ** 2
                )
            )
        ),
    )

    print()
    print(
        "PI05 TEACHER-FORCING "
        "DIAGNOSTIC PASSED"
    )


if __name__ == "__main__":
    main()
