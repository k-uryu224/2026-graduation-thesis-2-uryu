#!/usr/bin/env python3

from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

try:
    from lerobot.utils.control_utils import predict_action
except ImportError:
    from lerobot.common.control_utils import predict_action


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


def main() -> None:
    assert DATASET_ROOT.is_dir(), DATASET_ROOT
    assert CHECKPOINT.is_dir(), CHECKPOINT

    print("=== load dataset ===")

    dataset = LeRobotDataset(
        repo_id="local/so101_pick_object_train_60eps",
        root=DATASET_ROOT,
    )

    print("frames:", len(dataset))
    print("episodes:", dataset.meta.total_episodes)

    print("\n=== load policy ===")

    policy = SmolVLAPolicy.from_pretrained(str(CHECKPOINT))
    policy = policy.to("cuda")
    policy.eval()
    policy.reset()

    print("device:", next(policy.parameters()).device)

    print("\n=== load processors ===")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {
                "rename_map": RENAME_MAP,
            },
        },
    )

    preprocessor.reset()
    postprocessor.reset()

    sample = dataset[0]

    def image_tensor_to_raw_numpy(image: torch.Tensor):
        """LeRobot dataset CHW float image -> raw inference HWC uint8."""
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
        ),
    }

    print("\n=== raw observation ===")
    for key, value in observation.items():
        print(key, value.shape, value.dtype)

    print("\n=== inference ===")

    with torch.inference_mode():
        action = predict_action(
            observation=observation,
            policy=policy,
            device=torch.device("cuda"),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=TASK,
            robot_type="so101_mujoco",
        )

    print("action shape:", tuple(action.shape))
    print("action dtype:", action.dtype)
    print("action:", action.detach().cpu().numpy())

    assert action.numel() == 6
    assert torch.isfinite(action).all()

    print("\nSMOLVLA BASELINE INFERENCE PASSED")


if __name__ == "__main__":
    main()
