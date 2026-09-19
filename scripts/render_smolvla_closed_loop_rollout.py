#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import av
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts/evaluate_smolvla_action_response.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "action_response_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(BASE_SCRIPT)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rollout-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    rollout_dir = args.rollout_dir.resolve()

    metadata_path = rollout_dir / "metadata.json"
    npz_path = rollout_dir / "rollout.npz"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    saved = np.load(npz_path)

    applied_actions = saved[
        "applied_actions"
    ].astype(np.float32)

    base = load_base_module()

    gen = base.load_generator_module()
    physics = gen.load_physics_module()

    dataset_config = base.load_yaml(
        base.DATASET_CONFIG_PATH
    )

    task = base.load_yaml(
        base.TASK_CONFIG_PATH
    )["task"]

    variations = gen.load_variations()

    variation = next(
        v
        for v in variations
        if v["id"]
        == metadata["variation_id"]
    )

    model = mujoco.MjModel.from_xml_path(
        str(gen.SCENE_PATH)
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
        gen.actuator_id(
            model,
            name,
        )
        for name in physics.ARM_JOINT_NAMES
    ]

    gripper_actuator = gen.actuator_id(
        model,
        physics.GRIPPER_JOINT_NAME,
    )

    actuator_ids = (
        arm_actuators
        + [gripper_actuator]
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    # Reconstruct exactly the same initial state.
    base.build_raw_observation(
        gen=gen,
        physics=physics,
        model=model,
        data=data,
        renderer=renderer,
        dataset_config=dataset_config,
        task=task,
        variation=variation,
        object_x=float(
            metadata["object_x_m"]
        ),
    )

    initial_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    fps = float(metadata["fps"])

    physics_rate_hz = (
        1.0
        / float(model.opt.timestep)
    )

    output_path = (
        rollout_dir
        / "top_wrist_rollout.mp4"
    )

    container = av.open(
        str(output_path),
        mode="w",
    )

    stream = container.add_stream(
        "libx264",
        rate=int(round(fps)),
    )

    stream.width = width * 2
    stream.height = height
    stream.pix_fmt = "yuv420p"

    physics_step_count = 0
    lift_history = []

    try:
        for frame_index, action in enumerate(
            applied_actions
        ):
            # Render observation_t before applying action_t.
            frames = []

            for camera_key in [
                "top",
                "wrist",
            ]:
                camera = (
                    dataset_config[
                        "cameras"
                    ][camera_key]
                )

                renderer.update_scene(
                    data,
                    camera=camera[
                        "mujoco_name"
                    ],
                )

                frames.append(
                    renderer.render().copy()
                )

            combined = np.concatenate(
                frames,
                axis=1,
            )

            video_frame = (
                av.VideoFrame.from_ndarray(
                    combined,
                    format="rgb24",
                )
            )

            for packet in stream.encode(
                video_frame
            ):
                container.mux(packet)

            # Apply saved SmolVLA action.
            for actuator, value in zip(
                actuator_ids,
                action,
                strict=True,
            ):
                data.ctrl[
                    actuator
                ] = float(value)

            target_steps = round(
                (
                    frame_index
                    + 1
                )
                * physics_rate_hz
                / fps
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

            lift_history.append(
                float(
                    data.site_xpos[
                        object_site_id,
                        2,
                    ]
                    - initial_object_z
                )
            )

        for packet in stream.encode():
            container.mux(packet)

    finally:
        container.close()
        renderer.close()

    lift_history = np.asarray(
        lift_history
    )

    print("=== replay result ===")
    print(
        "frames:",
        len(applied_actions),
    )
    print(
        "duration:",
        f"{len(applied_actions) / fps:.3f} s",
    )
    print(
        "final lift:",
        f"{lift_history[-1] * 1000:.2f} mm",
    )
    print(
        "max lift:",
        f"{lift_history.max() * 1000:.2f} mm",
    )

    expected_final = (
        float(
            metadata["final_lift_m"]
        )
        * 1000.0
    )

    actual_final = (
        lift_history[-1]
        * 1000.0
    )

    print(
        "saved rollout final:",
        f"{expected_final:.2f} mm",
    )
    print(
        "replay difference:",
        f"{abs(actual_final - expected_final):.6f} mm",
    )

    print()
    print("saved:", output_path)
    print()
    print("SMOLVLA ROLLOUT VIDEO PASSED")


if __name__ == "__main__":
    main()
