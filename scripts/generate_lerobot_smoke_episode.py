#!/usr/bin/env python3

from __future__ import annotations

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

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

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

TASK_PATH = (
    ROOT / "configs/task/pick_object.yaml"
)

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

DATASET_ROOT = ROOT / (
    "results/lerobot_datasets/"
    "so101_lprd_smoke_x240_v00"
)

REPO_ID = (
    "local/so101_lprd_smoke_x240_v00"
)

OBJECT_POSITION = np.array(
    [0.240, 0.0, 0.065],
    dtype=float,
)

VARIATION_ID = "v00"

OPEN_CTRL = 1.0
CLOSED_CTRL = 0.0


def load_physics_module():
    spec = importlib.util.spec_from_file_location(
        "canonical_grasp_physics",
        PHYSICS_CHECK_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load canonical physics module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def load_variation():
    with VARIATIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        fixture = json.load(f)

    matches = [
        v
        for v in fixture["variations"]
        if v["id"] == VARIATION_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {VARIATION_ID}, "
            f"found {len(matches)}."
        )

    return matches[0]


def get_actuator_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
        )

    return actuator_id


def get_joint_qpos_address(
    model: mujoco.MjModel,
    name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return int(
        model.jnt_qposadr[joint_id]
    )


def get_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
) -> np.ndarray:
    values = []

    for joint_name in joint_names:
        qpos_address = (
            get_joint_qpos_address(
                model,
                joint_name,
            )
        )

        values.append(
            float(
                data.qpos[
                    qpos_address
                ]
            )
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def main() -> None:
    physics = load_physics_module()

    with TASK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        task_config = yaml.safe_load(f)

    task = task_config["task"]

    dataset_config = (
        load_lerobot_dataset_config(
            DATASET_CONFIG_PATH
        )
    )

    variation = load_variation()

    home_offsets_deg = np.asarray(
        variation[
            "home_joint_offsets_deg"
        ],
        dtype=float,
    )

    home_offsets_rad = np.deg2rad(
        home_offsets_deg
    )

    pregrasp_z_offset_m = float(
        variation[
            "pregrasp_z_offset_m"
        ]
    )

    print("=== experiment ===")
    print("object x:      ", OBJECT_POSITION[0])
    print("variation:     ", VARIATION_ID)
    print(
        "home offset deg:",
        home_offsets_deg.tolist(),
    )
    print(
        "pregrasp dz:   ",
        pregrasp_z_offset_m,
    )
    print("task:          ", task["instruction"])
    print()

    model = mujoco.MjModel.from_xml_path(
        str(SCENE_PATH)
    )

    data = mujoco.MjData(model)

    mujoco.mj_resetData(
        model,
        data,
    )

    initial_qpos = data.qpos.copy()

    physics.set_object_position(
        model,
        initial_qpos,
        OBJECT_POSITION,
    )

    # Fixed v00 home perturbation.
    for (
        joint_name,
        offset_rad,
    ) in zip(
        physics.ARM_JOINT_NAMES,
        home_offsets_rad,
        strict=True,
    ):
        qpos_address = (
            get_joint_qpos_address(
                model,
                joint_name,
            )
        )

        initial_qpos[
            qpos_address
        ] += float(offset_rad)

    # Canonical mapping:
    # 1.0 = open, 0.0 = closed.
    gripper_qpos_address = (
        get_joint_qpos_address(
            model,
            physics.GRIPPER_JOINT_NAME,
        )
    )

    initial_qpos[
        gripper_qpos_address
    ] = OPEN_CTRL

    trajectory = (
        physics.build_waypoint_trajectory(
            model=model,
            initial_qpos=initial_qpos,
            object_position=OBJECT_POSITION,
            task=task,
            pre_grasp_z_offset_m=(
                pregrasp_z_offset_m
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

    arm_actuators = [
        get_actuator_id(
            model,
            joint_name,
        )
        for joint_name
        in physics.ARM_JOINT_NAMES
    ]

    gripper_actuator = (
        get_actuator_id(
            model,
            physics.GRIPPER_JOINT_NAME,
        )
    )

    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0

    if model.na > 0:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    for actuator_id, value in zip(
        arm_actuators,
        trajectory.arm_qpos[0],
        strict=True,
    ):
        data.ctrl[
            actuator_id
        ] = float(value)

    data.ctrl[
        gripper_actuator
    ] = OPEN_CTRL

    mujoco.mj_forward(
        model,
        data,
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    initial_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    height = int(
        dataset_config[
            "images"
        ][
            "height"
        ]
    )

    width = int(
        dataset_config[
            "images"
        ][
            "width"
        ]
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    # --------------------------------------------------
    # Create new LeRobot dataset.
    # --------------------------------------------------

    if DATASET_ROOT.exists():
        shutil.rmtree(
            DATASET_ROOT
        )

    features = build_lerobot_features(
        dataset_config
    )

    print()
    print("=== LeRobot dataset ===")
    print("root:   ", DATASET_ROOT)
    print("repo_id:", REPO_ID)
    print("fps:    ", trajectory.rate_hz)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=int(trajectory.rate_hz),
        features=features,
        root=DATASET_ROOT,
        robot_type=dataset_config[
            "robot_type"
        ],
        use_videos=bool(
            dataset_config[
                "images"
            ][
                "use_videos"
            ]
        ),
    )

    physics_rate_hz = (
        1.0
        / float(
            model.opt.timestep
        )
    )

    physics_step_count = 0

    joint_names = list(
        dataset_config[
            "state"
        ][
            "mujoco_joints"
        ]
    )

    try:
        for frame_index in range(
            trajectory.frame_count
        ):
            # ==========================================
            # Observation at time t.
            # ==========================================

            frame: dict[str, object] = {}

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
                    camera=camera_name,
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

            # ==========================================
            # Canonical action for interval t -> t+1.
            # ==========================================

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
                        [gripper_target],
                        dtype=np.float32,
                    ),
                ]
            ).astype(
                np.float32,
                copy=False,
            )

            if state.shape != (6,):
                raise RuntimeError(
                    f"state shape: "
                    f"{state.shape}"
                )

            if action.shape != (6,):
                raise RuntimeError(
                    f"action shape: "
                    f"{action.shape}"
                )

            frame["action"] = action
            frame["task"] = task[
                "instruction"
            ]

            dataset.add_frame(
                frame
            )

            # ==========================================
            # Apply action and advance exactly to the
            # next 30 Hz observation timestamp.
            # ==========================================

            for (
                actuator_id,
                value,
            ) in zip(
                arm_actuators,
                arm_target,
                strict=True,
            ):
                data.ctrl[
                    actuator_id
                ] = float(value)

            data.ctrl[
                gripper_actuator
            ] = float(
                gripper_target
            )

            target_physics_steps = round(
                (
                    frame_index
                    + 1
                )
                * physics_rate_hz
                / trajectory.rate_hz
            )

            while (
                physics_step_count
                < target_physics_steps
            ):
                mujoco.mj_step(
                    model,
                    data,
                )

                physics_step_count += 1

            if (
                frame_index % 30 == 0
                or frame_index
                == trajectory.frame_count - 1
            ):
                print(
                    f"frame "
                    f"{frame_index + 1:3d}"
                    f"/{trajectory.frame_count}"
                    f"  sim="
                    f"{data.time:.3f}s"
                )

    finally:
        renderer.close()

    # One episode only.
    dataset.save_episode(
        parallel_encoding=False
    )

    dataset.finalize()

    final_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    lift_mm = (
        final_object_z
        - initial_object_z
    ) * 1000.0

    print()
    print("=== simulation result ===")
    print(
        "simulation time:",
        f"{data.time:.4f} s",
    )
    print(
        "object lift:",
        f"{lift_mm:.2f} mm",
    )

    print()
    print(
        "DATASET GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()
