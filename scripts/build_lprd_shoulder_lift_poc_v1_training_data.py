#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs/training/"
    "lprd_shoulder_lift_poc_v1_training_data.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/lprd_shoulder_lift_poc_v1/"
    "training_data"
)

EXTRACTOR_PATH = (
    ROOT
    / "scripts/"
    "extract_pi05_shoulder_lift_response_poc_v1.py"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def load_module(
    path: Path,
    name: str,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[name] = module
    spec.loader.exec_module(
        module
    )

    return module


def offset_key(
    delta_deg: float,
):
    magnitude = int(
        round(
            abs(
                float(delta_deg)
            )
        )
    )

    prefix = (
        "m"
        if delta_deg < 0
        else "p"
    )

    return (
        f"{prefix}"
        f"{magnitude:03d}"
    )


def check_in_range(
    actions,
    ctrl_ranges,
    *,
    name,
):
    low = (
        ctrl_ranges[:, 0]
        .reshape(1, 6)
    )

    high = (
        ctrl_ranges[:, 1]
        .reshape(1, 6)
    )

    if np.any(
        actions < low - 1e-6
    ):
        raise RuntimeError(
            f"{name}: below actuator range"
        )

    if np.any(
        actions > high + 1e-6
    ):
        raise RuntimeError(
            f"{name}: above actuator range"
        )


def main():
    args = parse_args()

    config_path = (
        args.config.resolve()
    )

    config = load_json(
        config_path
    )

    response_path = (
        ROOT
        / config[
            "teacher_response_path"
        ]
    ).resolve()

    response_lock = (
        ROOT
        / config[
            "teacher_response_lock"
        ]
    ).resolve()

    if not response_path.is_file():
        raise FileNotFoundError(
            response_path
        )

    if not response_lock.is_file():
        raise FileNotFoundError(
            response_lock
        )

    # Verify that frozen Teacher Response
    # files have not changed.
    subprocess.run(
        [
            "sha256sum",
            "-c",
            str(response_lock),
        ],
        cwd=ROOT,
        check=True,
    )

    responses = np.load(
        response_path,
        allow_pickle=False,
    )

    variation_ids = list(
        config[
            "variation_ids"
        ]
    )

    offsets = [
        float(x)
        for x in config[
            "training_offsets_deg"
        ]
    ]

    heldout = float(
        config[
            "heldout_evaluation_offset_deg"
        ]
    )

    if any(
        math.isclose(
            delta,
            heldout,
            abs_tol=1e-12,
        )
        for delta in offsets
    ):
        raise RuntimeError(
            "Held-out -5deg appears "
            "in training offsets."
        )

    if variation_ids != [
        "v00",
        "v01",
        "v02",
        "v03",
        "v04",
    ]:
        raise RuntimeError(
            "Expected exactly v00-v04."
        )

    if sorted(offsets) != [
        -6.0,
        -4.0,
    ]:
        raise RuntimeError(
            "Expected exactly "
            "[-6,-4] training offsets."
        )

    stored_variations = (
        responses[
            "variation_ids"
        ]
        .tolist()
    )

    if stored_variations != (
        variation_ids
    ):
        raise RuntimeError(
            "Variation order mismatch "
            "with frozen Teacher Response."
        )

    demo = (
        responses[
            "demo_action_ref"
        ]
        .astype(np.float32)
    )

    ctrl_ranges = (
        responses[
            "actuator_ctrl_ranges"
        ]
        .astype(np.float32)
    )

    if demo.shape != (
        5,
        50,
        6,
    ):
        raise RuntimeError(
            f"Unexpected demo shape: "
            f"{demo.shape}"
        )

    if ctrl_ranges.shape != (
        6,
        2,
    ):
        raise RuntimeError(
            "Unexpected actuator range shape."
        )

    required_keys = []

    for delta in offsets:
        key = offset_key(
            delta
        )

        required_keys.extend(
            [
                f"teacher_action_{key}_mean",
                f"response_target_preclip_{key}",
                f"response_target_{key}",
                f"response_target_clip_mask_{key}",
            ]
        )

    missing = [
        key
        for key in required_keys
        if key not in responses.files
    ]

    if missing:
        raise RuntimeError(
            f"Missing frozen response arrays: "
            f"{missing}"
        )

    print(
        "=== training data validation ==="
    )

    print(
        "variations:",
        variation_ids,
    )

    print(
        "offsets:",
        offsets,
    )

    print(
        "heldout:",
        heldout,
    )

    print(
        "sample_count:",
        len(variation_ids)
        * len(offsets),
    )

    print(
        "demo:",
        demo.shape,
    )

    print(
        "ctrl_ranges:",
        ctrl_ranges.shape,
    )

    if args.validate_only:
        print(
            "TRAINING DATA "
            "VALIDATION ONLY PASSED"
        )
        return

    output_dir = (
        args.output_dir.resolve()
    )

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(
                output_dir
            )
        else:
            raise SystemExit(
                f"Output exists:\n"
                f"{output_dir}\n"
                "Use --overwrite only "
                "if intentional."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extractor = load_module(
        EXTRACTOR_PATH,
        "lprd_teacher_response_extractor",
    )

    base = (
        extractor.load_base_module()
    )

    gen = (
        base.load_generator_module()
    )

    physics = (
        gen.load_physics_module()
    )

    dataset_config = (
        base.load_yaml(
            base.DATASET_CONFIG_PATH
        )
    )

    task = (
        base.load_yaml(
            base.TASK_CONFIG_PATH
        )[
            "task"
        ]
    )

    all_variations = {
        item["id"]: item
        for item in gen.load_variations()
    }

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                gen.SCENE_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    renderer = mujoco.Renderer(
        model,
        height=int(
            dataset_config[
                "images"
            ][
                "height"
            ]
        ),
        width=int(
            dataset_config[
                "images"
            ][
                "width"
            ]
        ),
    )

    object_site_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target_box_center",
        )
    )

    if object_site_id < 0:
        raise RuntimeError(
            "target_box_center "
            "site not found."
        )

    states = []
    top_images = []
    wrist_images = []

    targets_no_response = []
    targets_pointwise = []
    targets_pointwise_preclip = []
    pointwise_clip_masks = []

    targets_response = []
    targets_response_preclip = []
    response_clip_masks = []

    sample_variation_ids = []
    sample_offsets = []
    sample_episode_indices = []

    manifest_rows = []

    source_episode_indices = (
        responses[
            "episode_indices"
        ]
        .astype(np.int64)
    )

    object_x = float(
        config[
            "object_x_m"
        ]
    )

    low = (
        ctrl_ranges[:, 0]
        .reshape(1, 6)
    )

    high = (
        ctrl_ranges[:, 1]
        .reshape(1, 6)
    )

    try:
        sample_index = 0

        for variation_index, variation_id in enumerate(
            variation_ids
        ):
            variation = (
                all_variations[
                    variation_id
                ]
            )

            for delta in offsets:
                key = offset_key(
                    delta
                )

                observation = (
                    extractor
                    .build_observation_at_offset(
                        base=base,
                        gen=gen,
                        physics=physics,
                        model=model,
                        data=data,
                        renderer=renderer,
                        dataset_config=(
                            dataset_config
                        ),
                        task=task,
                        variation=variation,
                        object_x=object_x,
                        shoulder_lift_offset_deg=(
                            delta
                        ),
                        object_site_id=(
                            object_site_id
                        ),
                    )
                )

                state = (
                    observation[
                        "observation.state"
                    ]
                    .copy()
                    .astype(np.float32)
                )

                top = (
                    observation[
                        "observation.images.top"
                    ]
                    .copy()
                )

                wrist = (
                    observation[
                        "observation.images.wrist"
                    ]
                    .copy()
                )

                no_response = (
                    demo[
                        variation_index
                    ]
                    .copy()
                )

                pointwise_preclip = (
                    responses[
                        f"teacher_action_{key}_mean"
                    ][
                        variation_index
                    ]
                    .copy()
                    .astype(np.float32)
                )

                pointwise = np.clip(
                    pointwise_preclip,
                    low,
                    high,
                ).astype(np.float32)

                pointwise_mask = (
                    np.abs(
                        pointwise_preclip
                        - pointwise
                    )
                    > 1e-9
                )

                response_preclip = (
                    responses[
                        f"response_target_preclip_{key}"
                    ][
                        variation_index
                    ]
                    .copy()
                    .astype(np.float32)
                )

                response_target = (
                    responses[
                        f"response_target_{key}"
                    ][
                        variation_index
                    ]
                    .copy()
                    .astype(np.float32)
                )

                response_mask = (
                    responses[
                        f"response_target_clip_mask_{key}"
                    ][
                        variation_index
                    ]
                    .copy()
                )

                for name, target in [
                    (
                        "no_response",
                        no_response,
                    ),
                    (
                        "pointwise_kd",
                        pointwise,
                    ),
                    (
                        "response_target",
                        response_target,
                    ),
                ]:
                    if target.shape != (
                        50,
                        6,
                    ):
                        raise RuntimeError(
                            f"{variation_id} "
                            f"{delta:+.1f} "
                            f"{name}: "
                            f"shape={target.shape}"
                        )

                    check_in_range(
                        target,
                        ctrl_ranges,
                        name=(
                            f"{variation_id} "
                            f"{delta:+.1f} "
                            f"{name}"
                        ),
                    )

                states.append(
                    state
                )

                top_images.append(
                    top
                )

                wrist_images.append(
                    wrist
                )

                targets_no_response.append(
                    no_response
                )

                targets_pointwise_preclip.append(
                    pointwise_preclip
                )

                targets_pointwise.append(
                    pointwise
                )

                pointwise_clip_masks.append(
                    pointwise_mask
                )

                targets_response_preclip.append(
                    response_preclip
                )

                targets_response.append(
                    response_target
                )

                response_clip_masks.append(
                    response_mask
                )

                sample_variation_ids.append(
                    variation_id
                )

                sample_offsets.append(
                    delta
                )

                episode_index = int(
                    source_episode_indices[
                        variation_index
                    ]
                )

                sample_episode_indices.append(
                    episode_index
                )

                manifest_rows.append(
                    {
                        "sample_index":
                            sample_index,
                        "variation_id":
                            variation_id,
                        "source_episode_index":
                            episode_index,
                        "shoulder_lift_offset_deg":
                            delta,
                        "object_x_m":
                            object_x,
                        "pointwise_clip_count":
                            int(
                                np.count_nonzero(
                                    pointwise_mask
                                )
                            ),
                        "response_clip_count":
                            int(
                                np.count_nonzero(
                                    response_mask
                                )
                            ),
                    }
                )

                print(
                    f"sample "
                    f"{sample_index:02d} "
                    f"{variation_id} "
                    f"dlift={delta:+.1f}deg "
                    f"point_clip="
                    f"{np.count_nonzero(pointwise_mask)} "
                    f"response_clip="
                    f"{np.count_nonzero(response_mask)}"
                )

                sample_index += 1

    finally:
        renderer.close()

    arrays = {
        "variation_ids":
            np.asarray(
                sample_variation_ids
            ),
        "shoulder_lift_offsets_deg":
            np.asarray(
                sample_offsets,
                dtype=np.float32,
            ),
        "source_episode_indices":
            np.asarray(
                sample_episode_indices,
                dtype=np.int64,
            ),
        "observation_state":
            np.stack(
                states
            ).astype(np.float32),
        "observation_images_top":
            np.stack(
                top_images
            ),
        "observation_images_wrist":
            np.stack(
                wrist_images
            ),
        "target_no_response":
            np.stack(
                targets_no_response
            ).astype(np.float32),
        "target_pointwise_kd_preclip":
            np.stack(
                targets_pointwise_preclip
            ).astype(np.float32),
        "target_pointwise_kd":
            np.stack(
                targets_pointwise
            ).astype(np.float32),
        "target_pointwise_kd_clip_mask":
            np.stack(
                pointwise_clip_masks
            ),
        "target_response_preclip":
            np.stack(
                targets_response_preclip
            ).astype(np.float32),
        "target_response":
            np.stack(
                targets_response
            ).astype(np.float32),
        "target_response_clip_mask":
            np.stack(
                response_clip_masks
            ),
        "actuator_ctrl_ranges":
            ctrl_ranges,
    }

    expected_sample_count = int(
        config[
            "sample_count"
        ]
    )

    if arrays[
        "observation_state"
    ].shape != (
        expected_sample_count,
        6,
    ):
        raise RuntimeError(
            "Unexpected state shape."
        )

    for key in [
        "target_no_response",
        "target_pointwise_kd",
        "target_response",
    ]:
        if arrays[
            key
        ].shape != (
            expected_sample_count,
            50,
            6,
        ):
            raise RuntimeError(
                f"{key}: "
                f"{arrays[key].shape}"
            )

    np.savez_compressed(
        output_dir
        / "training_samples.npz",
        **arrays,
    )

    with (
        output_dir
        / "sample_manifest.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                manifest_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    metadata = {
        "dataset_id":
            config[
                "dataset_id"
            ],
        "source_git_commit":
            git_commit(),
        "config_path":
            str(
                config_path
            ),
        "config_sha256":
            sha256_file(
                config_path
            ),
        "teacher_response_path":
            str(
                response_path
            ),
        "teacher_response_sha256":
            sha256_file(
                response_path
            ),
        "teacher_response_lock":
            str(
                response_lock
            ),
        "teacher_response_lock_sha256":
            sha256_file(
                response_lock
            ),
        "extractor_path":
            str(
                EXTRACTOR_PATH
            ),
        "extractor_sha256":
            sha256_file(
                EXTRACTOR_PATH
            ),
        "object_x_m":
            object_x,
        "variation_ids":
            variation_ids,
        "training_offsets_deg":
            offsets,
        "heldout_evaluation_offset_deg":
            heldout,
        "heldout_minus5_present":
            bool(
                np.any(
                    np.isclose(
                        arrays[
                            "shoulder_lift_offsets_deg"
                        ],
                        heldout,
                    )
                )
            ),
        "instruction":
            config[
                "instruction"
            ],
        "robot_type":
            config[
                "robot_type"
            ],
        "sample_count":
            expected_sample_count,
        "output_shapes": {
            key: list(
                value.shape
            )
            for key, value
            in arrays.items()
        },
        "total_pointwise_clip_count":
            int(
                np.count_nonzero(
                    arrays[
                        "target_pointwise_kd_clip_mask"
                    ]
                )
            ),
        "total_response_clip_count":
            int(
                np.count_nonzero(
                    arrays[
                        "target_response_clip_mask"
                    ]
                )
            ),
    }

    if metadata[
        "heldout_minus5_present"
    ]:
        raise RuntimeError(
            "Held-out -5deg leaked "
            "into training data."
        )

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")

    print()
    print(
        "TRAINING DATA BUILD PASSED"
    )

    print(
        "samples:",
        expected_sample_count,
    )

    print(
        "state:",
        arrays[
            "observation_state"
        ].shape,
    )

    print(
        "top:",
        arrays[
            "observation_images_top"
        ].shape,
    )

    print(
        "wrist:",
        arrays[
            "observation_images_wrist"
        ].shape,
    )

    print(
        "targets:",
        arrays[
            "target_response"
        ].shape,
    )

    print(
        "pointwise clip count:",
        metadata[
            "total_pointwise_clip_count"
        ],
    )

    print(
        "response clip count:",
        metadata[
            "total_response_clip_count"
        ],
    )

    print(
        "heldout -5 present:",
        metadata[
            "heldout_minus5_present"
        ],
    )


if __name__ == "__main__":
    main()
