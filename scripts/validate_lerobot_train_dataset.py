#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROOT = Path(
    "results/lerobot_datasets/"
    "so101_pick_object_train_60eps"
)

REPO_ID = "local/so101_pick_object_train_60eps"

EXPECTED_X = {
    0.220,
    0.240,
    0.260,
}

EXPECTED_VARIATIONS = {
    f"v{i:02d}"
    for i in range(20)
}

EXPECTED_EPISODES = 60
EXPECTED_FRAMES_PER_EPISODE = 210
EXPECTED_TOTAL_FRAMES = (
    EXPECTED_EPISODES
    * EXPECTED_FRAMES_PER_EPISODE
)

EXPECTED_VIDEO_KEYS = {
    "observation.images.top",
    "observation.images.wrist",
}


def main() -> None:
    print("=== metadata ===")

    info_path = ROOT / "meta/info.json"

    info = json.loads(
        info_path.read_text()
    )

    print(
        "episodes:",
        info["total_episodes"],
    )
    print(
        "frames:",
        info["total_frames"],
    )
    print(
        "fps:",
        info["fps"],
    )
    print(
        "robot_type:",
        info["robot_type"],
    )

    assert (
        info["total_episodes"]
        == EXPECTED_EPISODES
    )

    assert (
        info["total_frames"]
        == EXPECTED_TOTAL_FRAMES
    )

    assert info["fps"] == 30

    for key in EXPECTED_VIDEO_KEYS:
        assert (
            info["features"][key]["dtype"]
            == "video"
        )

    print()
    print("=== manifest ===")

    manifest = pd.read_csv(
        ROOT / "experiment_manifest.csv"
    )

    print(
        "rows:",
        len(manifest),
    )

    assert len(manifest) == 60
    assert manifest["episode_index"].nunique() == 60

    x_counts = (
        manifest["object_x_m"]
        .round(3)
        .value_counts()
        .sort_index()
    )

    print()
    print("episodes per x:")
    print(x_counts.to_string())

    assert set(x_counts.index) == EXPECTED_X
    assert (x_counts == 20).all()

    variation_counts = (
        manifest["variation_id"]
        .value_counts()
        .sort_index()
    )

    print()
    print("episodes per variation:")
    print(variation_counts.to_string())

    assert (
        set(variation_counts.index)
        == EXPECTED_VARIATIONS
    )

    assert (
        variation_counts == 3
    ).all()

    pairs = manifest[
        [
            "object_x_m",
            "variation_id",
        ]
    ].copy()

    pairs["object_x_m"] = (
        pairs["object_x_m"]
        .round(3)
    )

    assert (
        pairs.drop_duplicates().shape[0]
        == 60
    )

    assert (
        manifest["frame_count"]
        == 210
    ).all()

    assert (
        manifest["fps"]
        == 30
    ).all()

    print()
    print("lift statistics [mm]:")
    print(
        manifest[
            "final_lift_mm"
        ].describe().to_string()
    )

    # ----------------------------------------------
    # Parquet temporal structure
    # ----------------------------------------------

    print()
    print("=== parquet temporal structure ===")

    table = pads.dataset(
        ROOT / "data",
        format="parquet",
    ).to_table(
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
        ]
    )

    df = table.to_pandas()

    print(
        "parquet rows:",
        len(df),
    )

    assert (
        len(df)
        == EXPECTED_TOTAL_FRAMES
    )

    groups = df.groupby(
        "episode_index",
        sort=True,
    )

    assert groups.ngroups == 60

    for episode_index, group in groups:
        group = group.sort_values(
            "frame_index"
        )

        assert len(group) == 210

        assert (
            int(
                group["frame_index"].iloc[0]
            )
            == 0
        )

        assert (
            int(
                group["frame_index"].iloc[-1]
            )
            == 209
        )

        first_ts = float(
            group["timestamp"].iloc[0]
        )

        last_ts = float(
            group["timestamp"].iloc[-1]
        )

        assert abs(
            first_ts - 0.0
        ) < 1e-5

        assert abs(
            last_ts - (209 / 30)
        ) < 1e-4

    print(
        "all episodes: "
        "210 frames, "
        "timestamp 0 -> 6.9667 s"
    )

    # ----------------------------------------------
    # Video files
    # ----------------------------------------------

    print()
    print("=== videos ===")

    for key in sorted(
        EXPECTED_VIDEO_KEYS
    ):
        path = ROOT / "videos" / key

        files = sorted(
            path.rglob("*.mp4")
        )

        total_bytes = sum(
            f.stat().st_size
            for f in files
        )

        print(
            key,
            "files=",
            len(files),
            "size_MB=",
            f"{total_bytes / 1024**2:.2f}",
        )

        assert len(files) > 0
        assert total_bytes > 0

    # ----------------------------------------------
    # LeRobot reload + actual video decode
    # ----------------------------------------------

    print()
    print("=== LeRobot reload ===")

    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=ROOT,
    )

    print("len:", len(dataset))
    print(
        "video_keys:",
        dataset.meta.video_keys,
    )

    assert len(dataset) == 12600
    assert dataset.meta.total_episodes == 60
    assert dataset.meta.total_frames == 12600
    assert dataset.meta.fps == 30

    assert (
        set(dataset.meta.video_keys)
        == EXPECTED_VIDEO_KEYS
    )

    # Decode representative frames:
    # first episode, middle position, final episode.
    sample_indices = [
        0,
        209,
        20 * 210,
        40 * 210,
        59 * 210,
        60 * 210 - 1,
    ]

    for index in sample_indices:
        frame = dataset[index]

        top = frame[
            "observation.images.top"
        ]

        wrist = frame[
            "observation.images.wrist"
        ]

        state = frame[
            "observation.state"
        ]

        action = frame["action"]

        assert tuple(top.shape) == (
            3,
            480,
            640,
        )

        assert tuple(wrist.shape) == (
            3,
            480,
            640,
        )

        assert tuple(state.shape) == (6,)
        assert tuple(action.shape) == (6,)

        print(
            f"index={index:5d} "
            f"episode="
            f"{int(frame['episode_index']):02d} "
            f"frame="
            f"{int(frame['frame_index']):03d} "
            "decode=PASS"
        )

    print()
    print("=" * 72)
    print(
        "TRAIN DATASET VALIDATION PASSED"
    )
    print(
        "60 episodes / "
        "12600 frames / "
        "3 x positions / "
        "20 fixed variations / "
        "top+wrist video"
    )


if __name__ == "__main__":
    main()
