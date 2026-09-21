#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_vla_response.dataset import (
    build_lerobot_features,
    load_lerobot_dataset_config,
)


ROOT = Path(__file__).resolve().parents[1]

SCENE_PATH = ROOT / (
    "assets/robots/so101/"
    "so_arm101_description/mjcf/"
    "top_grasp_scene.xml"
)

TASK_PATH = ROOT / "configs/task/pick_object.yaml"

VARIATIONS_PATH = ROOT / (
    "configs/trajectory/"
    "grasp_variations_v1.json"
)

DATASET_CONFIG_PATH = ROOT / (
    "configs/dataset/"
    "so101_top_wrist.yaml"
)

PHYSICS_CHECK_PATH = (
    ROOT / "scripts/check_grasp_physics.py"
)

DEFAULT_OUTPUT_ROOT = ROOT / (
    "results/lerobot_datasets/"
    "so101_pick_object_train_dense_x_100eps"
)

DEFAULT_REPO_ID = (
    "local/so101_pick_object_train_dense_x_100eps"
)

TRAIN_X_M = (
    0.220,
    0.230,
    0.240,
    0.250,
    0.260,
)

OBJECT_Y_M = 0.0
OBJECT_Z_M = 0.065

OPEN_CTRL = 1.0
CLOSED_CTRL = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_physics_module():
    spec = importlib.util.spec_from_file_location(
        "canonical_grasp_physics_train",
        PHYSICS_CHECK_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load canonical physics module."
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def load_variations() -> list[dict]:
    with VARIATIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        fixture = json.load(f)

    variations = fixture["variations"]

    expected_ids = [
        f"v{i:02d}"
        for i in range(20)
    ]

    actual_ids = [
        v["id"]
        for v in variations
    ]

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Variation fixture order/content differs "
            "from fixed v00-v19 fixture.\n"
            f"expected={expected_ids}\n"
            f"actual={actual_ids}"
        )

    return variations


def actuator_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
        )

    return idx


def qpos_address(
    model: mujoco.MjModel,
    joint_name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    return int(
        model.jnt_qposadr[joint_id]
    )


def get_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            float(
                data.qpos[
                    qpos_address(
                        model,
                        joint_name,
                    )
                ]
            )
            for joint_name in joint_names
        ],
        dtype=np.float32,
    )


def prepare_episode(
    *,
    physics,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: dict,
    object_position: np.ndarray,
    variation: dict,
):
    mujoco.mj_resetData(
        model,
        data,
    )

    initial_qpos = data.qpos.copy()

    physics.set_object_position(
        model,
        initial_qpos,
        object_position,
    )

    offsets_deg = np.asarray(
        variation["home_joint_offsets_deg"],
        dtype=float,
    )

    offsets_rad = np.deg2rad(
        offsets_deg
    )

    for (
        joint_name,
        offset_rad,
    ) in zip(
        physics.ARM_JOINT_NAMES,
        offsets_rad,
        strict=True,
    ):
        initial_qpos[
            qpos_address(
                model,
                joint_name,
            )
        ] += float(offset_rad)

    initial_qpos[
        qpos_address(
            model,
            physics.GRIPPER_JOINT_NAME,
        )
    ] = OPEN_CTRL

    trajectory = (
        physics.build_waypoint_trajectory(
            model=model,
            initial_qpos=initial_qpos,
            object_position=object_position,
            task=task,
            pre_grasp_z_offset_m=float(
                variation[
                    "pregrasp_z_offset_m"
                ]
            ),
        )
    )

    if trajectory.frame_count != 210:
        raise RuntimeError(
            f"Expected 210 frames, got "
            f"{trajectory.frame_count}."
        )

    if not math.isclose(
        trajectory.rate_hz,
        30.0,
    ):
        raise RuntimeError(
            f"Expected 30 Hz, got "
            f"{trajectory.rate_hz}."
        )

    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0

    if model.na > 0:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    return initial_qpos, trajectory


def main() -> None:
    args = parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists:\n"
                f"{args.output_root}\n\n"
                "Use --overwrite only if you intentionally "
                "want to regenerate it."
            )

        shutil.rmtree(
            args.output_root
        )

    physics = load_physics_module()

    with TASK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        task = yaml.safe_load(f)["task"]

    dataset_config = (
        load_lerobot_dataset_config(
            DATASET_CONFIG_PATH
        )
    )

    variations = load_variations()

    features = build_lerobot_features(
        dataset_config
    )

    model = mujoco.MjModel.from_xml_path(
        str(SCENE_PATH)
    )

    data = mujoco.MjData(model)

    height = int(
        dataset_config["images"]["height"]
    )

    width = int(
        dataset_config["images"]["width"]
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    arm_actuators = [
        actuator_id(
            model,
            joint_name,
        )
        for joint_name
        in physics.ARM_JOINT_NAMES
    ]

    gripper_actuator = actuator_id(
        model,
        physics.GRIPPER_JOINT_NAME,
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    if object_site_id < 0:
        raise RuntimeError(
            "target_box_center site not found."
        )

    joint_names = list(
        dataset_config[
            "state"
        ][
            "mujoco_joints"
        ]
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        features=features,
        root=args.output_root,
        robot_type=dataset_config[
            "robot_type"
        ],
        use_videos=True,
    )

    physics_rate_hz = (
        1.0
        / float(
            model.opt.timestep
        )
    )

    manifest_rows: list[dict] = []

    total_episodes = (
        len(TRAIN_X_M)
        * len(variations)
    )

    episode_index = 0

    try:
        for object_x in TRAIN_X_M:
            for variation in variations:
                variation_id = variation["id"]

                object_position = np.asarray(
                    [
                        object_x,
                        OBJECT_Y_M,
                        OBJECT_Z_M,
                    ],
                    dtype=float,
                )

                print()
                print("=" * 72)
                print(
                    f"episode "
                    f"{episode_index + 1:02d}"
                    f"/{total_episodes:02d}"
                    f"  x={object_x:.3f}"
                    f"  {variation_id}"
                )

                (
                    _,
                    trajectory,
                ) = prepare_episode(
                    physics=physics,
                    model=model,
                    data=data,
                    task=task,
                    object_position=(
                        object_position
                    ),
                    variation=variation,
                )

                home_arm = (
                    trajectory.arm_qpos[0]
                )

                for (
                    actuator,
                    value,
                ) in zip(
                    arm_actuators,
                    home_arm,
                    strict=True,
                ):
                    data.ctrl[
                        actuator
                    ] = float(value)

                data.ctrl[
                    gripper_actuator
                ] = OPEN_CTRL

                mujoco.mj_forward(
                    model,
                    data,
                )

                initial_object_z = float(
                    data.site_xpos[
                        object_site_id,
                        2,
                    ]
                )

                physics_step_count = 0

                for frame_index in range(
                    trajectory.frame_count
                ):
                    # ----------------------------------
                    # observation_t
                    # ----------------------------------

                    frame: dict[
                        str,
                        object,
                    ] = {}

                    state = get_state(
                        model,
                        data,
                        joint_names,
                    )

                    frame[
                        "observation.state"
                    ] = state

                    for camera in (
                        dataset_config[
                            "cameras"
                        ].values()
                    ):
                        camera_name = camera[
                            "mujoco_name"
                        ]

                        lerobot_key = camera[
                            "lerobot_key"
                        ]

                        renderer.update_scene(
                            data,
                            camera=(
                                camera_name
                            ),
                        )

                        image = (
                            renderer
                            .render()
                            .copy()
                        )

                        if image.shape != (
                            height,
                            width,
                            3,
                        ):
                            raise RuntimeError(
                                f"{camera_name}: "
                                f"unexpected image "
                                f"shape {image.shape}"
                            )

                        frame[
                            lerobot_key
                        ] = image

                    # ----------------------------------
                    # action_t
                    # ----------------------------------

                    arm_target = np.asarray(
                        trajectory.arm_qpos[
                            frame_index
                        ],
                        dtype=np.float32,
                    )

                    close_fraction = float(
                        trajectory
                        .gripper_close_fraction[
                            frame_index
                        ]
                    )

                    gripper_target = (
                        OPEN_CTRL
                        + close_fraction
                        * (
                            CLOSED_CTRL
                            - OPEN_CTRL
                        )
                    )

                    action = np.concatenate(
                        [
                            arm_target,
                            np.asarray(
                                [
                                    gripper_target
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    ).astype(
                        np.float32,
                        copy=False,
                    )

                    if state.shape != (6,):
                        raise RuntimeError(
                            f"state shape="
                            f"{state.shape}"
                        )

                    if action.shape != (6,):
                        raise RuntimeError(
                            f"action shape="
                            f"{action.shape}"
                        )

                    frame["action"] = action

                    frame["task"] = task[
                        "instruction"
                    ]

                    dataset.add_frame(
                        frame
                    )

                    # ----------------------------------
                    # Apply action_t.
                    # ----------------------------------

                    for (
                        actuator,
                        value,
                    ) in zip(
                        arm_actuators,
                        arm_target,
                        strict=True,
                    ):
                        data.ctrl[
                            actuator
                        ] = float(value)

                    data.ctrl[
                        gripper_actuator
                    ] = float(
                        gripper_target
                    )

                    target_steps = round(
                        (
                            frame_index
                            + 1
                        )
                        * physics_rate_hz
                        / trajectory.rate_hz
                    )

                    while (
                        physics_step_count
                        < target_steps
                    ):
                        mujoco.mj_step(
                            model,
                            data,
                        )

                        physics_step_count += 1

                if not math.isclose(
                    data.time,
                    7.0,
                    abs_tol=1e-8,
                ):
                    raise RuntimeError(
                        f"Episode ended at "
                        f"{data.time:.9f}s, "
                        "expected 7.0s."
                    )

                final_object_z = float(
                    data.site_xpos[
                        object_site_id,
                        2,
                    ]
                )

                lift_m = (
                    final_object_z
                    - initial_object_z
                )

                lift_mm = (
                    lift_m
                    * 1000.0
                )

                minimum_lift_m = float(
                    task[
                        "success"
                    ][
                        "minimum_lift_delta_m"
                    ]
                )

                if lift_m < minimum_lift_m:
                    raise RuntimeError(
                        f"{object_x=} "
                        f"{variation_id}: "
                        f"lift={lift_mm:.2f}mm "
                        f"< minimum "
                        f"{minimum_lift_m * 1000:.2f}mm"
                    )

                dataset.save_episode(
                    parallel_encoding=False
                )

                manifest_rows.append(
                    {
                        "episode_index":
                            episode_index,
                        "object_x_m":
                            object_x,
                        "object_y_m":
                            OBJECT_Y_M,
                        "object_z_m":
                            OBJECT_Z_M,
                        "variation_id":
                            variation_id,
                        "pregrasp_z_offset_m":
                            float(
                                variation[
                                    "pregrasp_z_offset_m"
                                ]
                            ),
                        "home_shoulder_pan_deg":
                            variation[
                                "home_joint_offsets_deg"
                            ][0],
                        "home_shoulder_lift_deg":
                            variation[
                                "home_joint_offsets_deg"
                            ][1],
                        "home_elbow_flex_deg":
                            variation[
                                "home_joint_offsets_deg"
                            ][2],
                        "home_wrist_flex_deg":
                            variation[
                                "home_joint_offsets_deg"
                            ][3],
                        "home_wrist_roll_deg":
                            variation[
                                "home_joint_offsets_deg"
                            ][4],
                        "final_lift_mm":
                            lift_mm,
                        "frame_count":
                            trajectory.frame_count,
                        "fps":
                            trajectory.rate_hz,
                    }
                )

                print(
                    f"PASS "
                    f"x={object_x:.3f} "
                    f"{variation_id} "
                    f"lift={lift_mm:.2f} mm"
                )

                episode_index += 1

    finally:
        renderer.close()

    dataset.finalize()

    manifest_path = (
        args.output_root
        / "experiment_manifest.csv"
    )

    with manifest_path.open(
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

    print()
    print("=" * 72)
    print("TRAIN DATASET GENERATION COMPLETE")
    print(
        "episodes:",
        len(manifest_rows),
    )
    print(
        "expected frames:",
        len(manifest_rows) * 210,
    )
    print(
        "root:",
        args.output_root,
    )
    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
