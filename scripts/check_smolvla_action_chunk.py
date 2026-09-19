#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

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
    / "outputs/train/smolvla_baseline_v1"
    / "checkpoints/030000/pretrained_model"
)

RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

TASK = "Pick up the object."
ROBOT_TYPE = "so101_mujoco"
NOISE_SEED = 20260920


def image_tensor_to_raw_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()

    assert image.ndim == 3, image.shape
    assert image.shape[0] in (1, 3, 4), image.shape

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


def build_processed_batch(
    *,
    sample,
    preprocessor,
    device,
):
    observation = {
        "observation.images.top": image_tensor_to_raw_numpy(
            sample["observation.images.top"]
        ),
        "observation.images.wrist": image_tensor_to_raw_numpy(
            sample["observation.images.wrist"]
        ),
        "observation.state": (
            sample["observation.state"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        ),
    }

    batch = prepare_observation_for_inference(
        observation,
        device,
        TASK,
        ROBOT_TYPE,
    )

    return preprocessor(batch)


def main() -> None:
    assert DATASET_ROOT.is_dir(), DATASET_ROOT
    assert CHECKPOINT.is_dir(), CHECKPOINT

    device = torch.device("cuda")

    print("=== load dataset ===")

    dataset = LeRobotDataset(
        repo_id="local/so101_pick_object_train_60eps",
        root=DATASET_ROOT,
    )

    sample = dataset[0]

    print("frames:", len(dataset))
    print("episodes:", dataset.meta.total_episodes)

    print("\n=== load policy ===")

    policy = SmolVLAPolicy.from_pretrained(
        str(CHECKPOINT)
    )
    policy = policy.to(device)
    policy.eval()
    policy.reset()

    print("chunk_size:", policy.config.chunk_size)
    print("n_action_steps:", policy.config.n_action_steps)
    print("max_action_dim:", policy.config.max_action_dim)
    print(
        "action_dim:",
        policy.config.action_feature.shape[0],
    )

    print("\n=== load processors ===")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
            "rename_observations_processor": {
                "rename_map": RENAME_MAP,
            },
        },
    )

    preprocessor.reset()
    postprocessor.reset()

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    generator = torch.Generator(
        device=device,
    )
    generator.manual_seed(NOISE_SEED)

    fixed_noise = torch.randn(
        noise_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    print("\n=== fixed noise ===")
    print("seed:", NOISE_SEED)
    print("shape:", tuple(fixed_noise.shape))
    print(
        "mean/std:",
        float(fixed_noise.mean()),
        float(fixed_noise.std()),
    )

    chunks = []

    for run_index in range(2):
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        batch = build_processed_batch(
            sample=sample,
            preprocessor=preprocessor,
            device=device,
        )

        with torch.inference_mode():
            raw_chunk = policy.predict_action_chunk(
                batch,
                noise=fixed_noise.clone(),
            )

            action_chunk = postprocessor(
                raw_chunk
            )

        chunks.append(
            action_chunk.detach().cpu()
        )

        print(
            f"\n=== run {run_index + 1} ==="
        )
        print(
            "raw chunk shape:",
            tuple(raw_chunk.shape),
        )
        print(
            "post chunk shape:",
            tuple(action_chunk.shape),
        )
        print(
            "first action:",
            action_chunk[0, 0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "last action:",
            action_chunk[0, -1]
            .detach()
            .cpu()
            .numpy(),
        )

    diff = torch.max(
        torch.abs(
            chunks[0] - chunks[1]
        )
    ).item()

    print("\n=== deterministic check ===")
    print("max abs diff:", diff)

    if chunks[0].shape != (
        1,
        policy.config.chunk_size,
        6,
    ):
        raise RuntimeError(
            f"Unexpected action chunk shape: "
            f"{tuple(chunks[0].shape)}"
        )

    if not torch.isfinite(
        chunks[0]
    ).all():
        raise RuntimeError(
            "Non-finite values in action chunk."
        )

    if diff > 1e-6:
        raise RuntimeError(
            f"Fixed-noise inference is not deterministic: {diff}"
        )

    print()
    print(
        "SMOLVLA FIXED-NOISE ACTION CHUNK PASSED"
    )


if __name__ == "__main__":
    main()
