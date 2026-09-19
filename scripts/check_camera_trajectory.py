#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw

from so101_vla_response.dataset import (
    load_lerobot_dataset_config,
)


ROOT = Path(__file__).resolve().parents[1]

SCENE_PATH = ROOT / (
    "assets/robots/so101/"
    "so_arm101_description/mjcf/"
    "top_grasp_scene.xml"
)

TASK_PATH = ROOT / "configs/task/pick_object.yaml"

DATASET_CONFIG_PATH = ROOT / (
    "configs/dataset/so101_top_wrist.yaml"
)

PHYSICS_CHECK_PATH = (
    ROOT / "scripts/check_grasp_physics.py"
)

OUTPUT_DIR = (
    ROOT / "results/camera_check/trajectory"
)

OBJECT_POSITION = np.array(
    [0.240, 0.000, 0.065],
    dtype=float,
)

OPEN_CTRL = 1.0
CLOSED_CTRL = 0.0


def load_physics_module():
    spec = importlib.util.spec_from_file_location(
        "canonical_grasp_physics",
        PHYSICS_CHECK_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Failed to load check_grasp_physics.py"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


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


def joint_qpos_address(
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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    physics = load_physics_module()

    with TASK_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        task_config = yaml.safe_load(f)

    def find_canonical_task(node):
        candidates = []

        def walk(value):
            if not isinstance(value, dict):
                return

            if (
                "tcp" in value
                and "trajectory" in value
            ):
                candidates.append(value)

            for child in value.values():
                walk(child)

        walk(node)

        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one task mapping "
                "containing both 'tcp' and 'trajectory', "
                f"found {len(candidates)}."
            )

        return candidates[0]

    task = find_canonical_task(
        task_config
    )

    print(
        "canonical task keys:",
        list(task.keys()),
    )

    dataset_config = (
        load_lerobot_dataset_config(
            DATASET_CONFIG_PATH
        )
    )

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

    gripper_qpos_address = (
        joint_qpos_address(
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
            pre_grasp_z_offset_m=0.0,
        )
    )

    phases = physics.phase_slices(
        trajectory.rate_hz
    )

    capture_frames = {
        0: "home",
        phases[
            "home_to_pre_grasp"
        ].stop - 1: "pre_grasp",
        phases[
            "pre_grasp_to_grasp"
        ].stop - 1: "grasp",
        phases[
            "grasp_to_lift"
        ].stop - 1: "lift",
    }

    print("=== capture frames ===")
    for frame, label in (
        capture_frames.items()
    ):
        print(
            f"{label:10s}: frame {frame}"
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

    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0

    if model.na > 0:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    for actuator, value in zip(
        arm_actuators,
        trajectory.arm_qpos[0],
        strict=True,
    ):
        data.ctrl[actuator] = value

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

    physics_rate_hz = (
        1.0 / float(model.opt.timestep)
    )

    physics_step_count = 0

    saved_images = {}

    try:
        for frame_index in range(
            trajectory.frame_count
        ):
            arm_target = (
                trajectory.arm_qpos[
                    frame_index
                ]
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

            for actuator, value in zip(
                arm_actuators,
                arm_target,
                strict=True,
            ):
                data.ctrl[
                    actuator
                ] = value

            data.ctrl[
                gripper_actuator
            ] = gripper_target

            target_physics_steps = round(
                (
                    frame_index + 1
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
                frame_index
                not in capture_frames
            ):
                continue

            label = capture_frames[
                frame_index
            ]

            for alias, camera in (
                dataset_config[
                    "cameras"
                ].items()
            ):
                camera_name = camera[
                    "mujoco_name"
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

                output_path = (
                    OUTPUT_DIR
                    / f"{alias}_{label}.png"
                )

                Image.fromarray(
                    image
                ).save(
                    output_path
                )

                saved_images[
                    (label, alias)
                ] = image

                print(
                    f"saved frame="
                    f"{frame_index:3d} "
                    f"{alias:5s} "
                    f"{label:10s} "
                    f"-> {output_path}"
                )

    finally:
        renderer.close()

    final_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    final_lift_mm = (
        final_object_z
        - initial_object_z
    ) * 1000.0

    print()
    print("=== trajectory ===")
    print(
        "frames:",
        trajectory.frame_count,
    )
    print(
        "rate:",
        trajectory.rate_hz,
        "Hz",
    )
    print(
        "simulation time:",
        f"{data.time:.4f}",
        "s",
    )
    print(
        "final object lift:",
        f"{final_lift_mm:.2f}",
        "mm",
    )

    # --------------------------------------------------
    # Contact sheet: 4 phases x 2 cameras
    # --------------------------------------------------

    labels = [
        "home",
        "pre_grasp",
        "grasp",
        "lift",
    ]

    aliases = [
        "top",
        "wrist",
    ]

    title_h = 30

    sheet = Image.new(
        "RGB",
        (
            width * 2,
            (height + title_h) * 4,
        ),
    )

    draw = ImageDraw.Draw(sheet)

    for row, label in enumerate(
        labels
    ):
        for col, alias in enumerate(
            aliases
        ):
            image = saved_images[
                (label, alias)
            ]

            x = col * width
            y = row * (
                height + title_h
            )

            draw.text(
                (
                    x + 10,
                    y + 8,
                ),
                f"{alias} / {label}",
                fill="white",
            )

            sheet.paste(
                Image.fromarray(image),
                (
                    x,
                    y + title_h,
                ),
            )

    sheet_path = (
        OUTPUT_DIR
        / "camera_contact_sheet.png"
    )

    sheet.save(
        sheet_path
    )

    print(
        "contact sheet:",
        sheet_path,
    )


if __name__ == "__main__":
    main()
