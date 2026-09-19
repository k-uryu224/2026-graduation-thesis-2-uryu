#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
from PIL import Image

from so101_vla_response.dataset import (
    load_lerobot_dataset_config,
)


DEFAULT_SCENE = Path(
    "assets/robots/so101/"
    "so_arm101_description/mjcf/"
    "top_grasp_scene.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/dataset/"
            "so101_top_wrist.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/camera_check"
        ),
    )
    parser.add_argument(
        "--label",
        default="home",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_lerobot_dataset_config(
        args.config
    )

    height = int(
        config["images"]["height"]
    )
    width = int(
        config["images"]["width"]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(args.scene)
    )
    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    print(
        f"resolution: {width}x{height}"
    )
    print(
        f"ncam: {model.ncam}"
    )

    try:
        for alias, camera in (
            config["cameras"].items()
        ):
            camera_name = camera[
                "mujoco_name"
            ]

            camera_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

            if camera_id < 0:
                raise RuntimeError(
                    f"MuJoCo camera not found: "
                    f"{camera_name}"
                )

            renderer.update_scene(
                data,
                camera=camera_name,
            )

            image = renderer.render()

            output = (
                args.output_dir
                / f"{alias}_{args.label}.png"
            )

            Image.fromarray(
                image
            ).save(output)

            print(
                f"{alias:8s}"
                f" mujoco={camera_name:8s}"
                f" shape={image.shape}"
                f" dtype={image.dtype}"
                f" -> {output}"
            )

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
