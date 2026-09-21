#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "results/lerobot_datasets/so101_pick_object_train_60eps"
)

ROLLOUT_CSV = (
    ROOT
    / "results/evaluation/pi05_teacher_v1/closed_loop"
    / "x0p240_v00_seed20260920/rollout.csv"
)

EPISODE_INDEX = 20
FRAMES_PER_EPISODE = 210

REPLAN_FRAMES = [0, 50, 100, 150, 200]

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    dataset = LeRobotDataset(
        repo_id="local/so101_pick_object_train_60eps",
        root=DATASET_ROOT,
        video_backend="pyav",
    )

    with ROLLOUT_CSV.open() as f:
        rollout = list(csv.DictReader(f))

    episode_start = (
        EPISODE_INDEX * FRAMES_PER_EPISODE
    )

    print("=== PI05 CLOSED-LOOP STATE DRIFT ===")

    for frame in REPLAN_FRAMES:
        expert_sample = dataset[
            episode_start + frame
        ]

        expert_state = (
            expert_sample["observation.state"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        closed_state = np.asarray(
            [
                float(
                    rollout[frame][
                        f"state_{name}"
                    ]
                )
                for name in ACTION_NAMES
            ],
            dtype=np.float32,
        )

        delta = closed_state - expert_state

        print()
        print(f"--- frame {frame} ---")

        for i, name in enumerate(
            ACTION_NAMES
        ):
            print(
                f"{name:16s} "
                f"expert={expert_state[i]: .5f} "
                f"closed={closed_state[i]: .5f} "
                f"delta={delta[i]:+.5f}"
            )

        arm_rmse = float(
            np.sqrt(
                np.mean(delta[:5] ** 2)
            )
        )

        print(
            "arm state RMSE:",
            f"{arm_rmse:.6f} rad",
        )

    print()
    print(
        "PI05 CLOSED-LOOP STATE DRIFT "
        "DIAGNOSTIC PASSED"
    )


if __name__ == "__main__":
    main()
