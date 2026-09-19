#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "PYOPENGL_PLATFORM",
    "egl",
)

import mujoco
import numpy as np

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

from so101_vla_response.dataset import (
    build_lerobot_features,
    load_lerobot_dataset_config,
)


CONFIG_PATH = Path(
    "configs/dataset/"
    "so101_top_wrist.yaml"
)

SCENE_PATH = Path(
    "assets/robots/so101/"
    "so_arm101_description/mjcf/"
    "top_grasp_scene.xml"
)


def get_joint_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
) -> np.ndarray:
    values = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise RuntimeError(
                f"Joint not found: {name}"
            )

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        values.append(
            data.qpos[qpos_address]
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def main() -> None:
    config = (
        load_lerobot_dataset_config(
            CONFIG_PATH
        )
    )

    features = build_lerobot_features(
        config
    )

    print("LeRobotDataset import: OK")
    print(LeRobotDataset)
    print()

    print("=== features ===")
    for key, value in features.items():
        print(
            key,
            value,
        )

    model = mujoco.MjModel.from_xml_path(
        str(SCENE_PATH)
    )
    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    height = int(
        config["images"]["height"]
    )
    width = int(
        config["images"]["width"]
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    frame: dict[str, object] = {}

    try:
        for camera in (
            config["cameras"].values()
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

            image = renderer.render()

            frame[lerobot_key] = (
                image.copy()
            )

    finally:
        renderer.close()

    state = get_joint_state(
        model,
        data,
        list(
            config["state"][
                "mujoco_joints"
            ]
        ),
    )

    # Schema smoke test only.
    # Real dataset generation will use
    # canonical Script+IK command here.
    action = state.copy()

    frame[
        "observation.state"
    ] = state

    frame["action"] = action

    frame["task"] = config["task"]

    print()
    print("=== frame ===")

    for key, value in frame.items():
        if isinstance(
            value,
            np.ndarray,
        ):
            print(
                f"{key:30s}"
                f" shape={value.shape}"
                f" dtype={value.dtype}"
            )
        else:
            print(
                f"{key:30s}"
                f" {value!r}"
            )

    for key, spec in features.items():
        value = frame[key]

        expected_shape = tuple(
            spec["shape"]
        )

        if value.shape != expected_shape:
            raise RuntimeError(
                f"{key}: shape "
                f"{value.shape} != "
                f"{expected_shape}"
            )

    if state.shape != (6,):
        raise RuntimeError(
            f"state shape: {state.shape}"
        )

    if action.shape != (6,):
        raise RuntimeError(
            f"action shape: {action.shape}"
        )

    print()
    print(
        "LEROBOT FRAME SCHEMA CHECK PASSED"
    )
    print(
        "Ready for LeRobotDataset.add_frame()."
    )


if __name__ == "__main__":
    main()
