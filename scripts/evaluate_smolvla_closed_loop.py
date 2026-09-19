#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts/evaluate_smolvla_action_response.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "smolvla_action_response_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {BASE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--object-x",
        type=float,
        default=0.240,
    )

    parser.add_argument(
        "--variation-id",
        default="v00",
    )

    parser.add_argument(
        "--noise-seed",
        type=int,
        default=20260920,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=210,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def render_observation(
    *,
    gen,
    physics,
    model,
    data,
    renderer,
    dataset_config,
):
    joint_names = list(
        dataset_config["state"]["mujoco_joints"]
    )

    state = gen.get_state(
        model,
        data,
        joint_names,
    ).copy()

    observation = {
        "observation.state":
            state.astype(np.float32),
    }

    for camera in dataset_config[
        "cameras"
    ].values():
        renderer.update_scene(
            data,
            camera=camera["mujoco_name"],
        )

        observation[
            camera["lerobot_key"]
        ] = (
            renderer.render()
            .copy()
        )

    return observation


def main():
    args = parse_args()

    base = load_base_module()

    run_name = (
        f"x{args.object_x:.3f}"
        f"_{args.variation_id}"
        f"_seed{args.noise_seed}"
    ).replace(".", "p")

    output_dir = (
        ROOT
        / "results/evaluation/smolvla_baseline_v1"
        / "closed_loop"
        / run_name
    )

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output already exists:\n"
                f"{output_dir}\n"
                f"Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda")

    # ----------------------------------------
    # Config / physics
    # ----------------------------------------

    gen = base.load_generator_module()
    physics = gen.load_physics_module()

    dataset_config = base.load_yaml(
        base.DATASET_CONFIG_PATH
    )

    task_doc = base.load_yaml(
        base.TASK_CONFIG_PATH
    )
    task = task_doc["task"]

    variations = gen.load_variations()

    variation = next(
        (
            item
            for item in variations
            if item["id"]
            == args.variation_id
        ),
        None,
    )

    if variation is None:
        raise RuntimeError(
            f"Variation not found: "
            f"{args.variation_id}"
        )

    fps = float(
        task["phases"]["timing"]["rate_hz"]
    )

    minimum_lift_m = float(
        task["success"][
            "minimum_lift_delta_m"
        ]
    )

    hold_duration_s = float(
        task["success"][
            "hold_duration_s"
        ]
    )

    hold_frames = int(
        math.ceil(
            hold_duration_s * fps
        )
    )

    # ----------------------------------------
    # Policy
    # ----------------------------------------

    print("=== load policy ===")

    policy = (
        base.SmolVLAPolicy
        .from_pretrained(
            str(base.CHECKPOINT)
        )
    )

    policy = policy.to(device)
    policy.eval()
    policy.reset()

    (
        preprocessor,
        postprocessor,
    ) = base.make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(
            base.CHECKPOINT
        ),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
            "rename_observations_processor": {
                "rename_map":
                    base.RENAME_MAP,
            },
        },
    )

    preprocessor.reset()
    postprocessor.reset()

    # Deterministic sequence of chunk noises.
    noise_generator = torch.Generator(
        device=device
    )

    noise_generator.manual_seed(
        args.noise_seed
    )

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    execution_horizon = min(
        policy.config.n_action_steps,
        policy.config.chunk_size,
    )

    print()
    print("=== rollout condition ===")
    print("object_x:", args.object_x)
    print("variation:", args.variation_id)
    print("noise_seed:", args.noise_seed)
    print("frames:", args.frames)
    print("fps:", fps)
    print(
        "chunk_size:",
        policy.config.chunk_size,
    )
    print(
        "execution_horizon:",
        execution_horizon,
    )
    print(
        "minimum_lift_m:",
        minimum_lift_m,
    )
    print(
        "hold_frames:",
        hold_frames,
    )

    # ----------------------------------------
    # MuJoCo
    # ----------------------------------------

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

    actuator_ids = np.asarray(
        arm_actuators
        + [gripper_actuator],
        dtype=int,
    )

    ctrl_ranges = np.asarray(
        [
            model.actuator_ctrlrange[
                actuator
            ]
            for actuator
            in actuator_ids
        ],
        dtype=np.float32,
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    if object_site_id < 0:
        raise RuntimeError(
            "target_box_center site not found"
        )

    physics_rate_hz = (
        1.0
        / float(
            model.opt.timestep
        )
    )

    # ----------------------------------------
    # Initial state
    # ----------------------------------------

    try:
        observation, _ = (
            base.build_raw_observation(
                gen=gen,
                physics=physics,
                model=model,
                data=data,
                renderer=renderer,
                dataset_config=dataset_config,
                task=task,
                variation=variation,
                object_x=args.object_x,
            )
        )

        initial_object_z = float(
            data.site_xpos[
                object_site_id,
                2,
            ]
        )

        initial_state = (
            observation[
                "observation.state"
            ]
            .copy()
        )

        print()
        print("initial state:", initial_state)
        print(
            "initial object z:",
            initial_object_z,
        )

        # ------------------------------------
        # Rollout
        # ------------------------------------

        active_chunk = None
        chunk_cursor = 0
        chunk_index = -1

        physics_step_count = 0

        consecutive_hold = 0
        maximum_hold = 0
        success_frame = None

        total_clipped_values = 0
        max_clip_delta = 0.0

        rows = []

        states_before = []
        raw_actions = []
        applied_actions = []
        object_z_history = []
        lift_history = []
        chunk_indices = []

        for frame_index in range(
            args.frames
        ):
            # --------------------------------
            # Replan when current chunk ends.
            # --------------------------------

            if (
                active_chunk is None
                or chunk_cursor
                >= execution_horizon
            ):
                if frame_index != 0:
                    observation = (
                        render_observation(
                            gen=gen,
                            physics=physics,
                            model=model,
                            data=data,
                            renderer=renderer,
                            dataset_config=(
                                dataset_config
                            ),
                        )
                    )

                batch = (
                    base.process_observation(
                        observation=observation,
                        preprocessor=preprocessor,
                        device=device,
                    )
                )

                chunk_noise = torch.randn(
                    noise_shape,
                    generator=noise_generator,
                    device=device,
                    dtype=torch.float32,
                )

                with torch.inference_mode():
                    raw_chunk = (
                        policy
                        .predict_action_chunk(
                            batch,
                            noise=chunk_noise,
                        )
                    )

                    processed_chunk = (
                        postprocessor(
                            raw_chunk.clone()
                        )
                    )

                active_chunk = (
                    processed_chunk[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                if active_chunk.shape != (
                    policy.config.chunk_size,
                    6,
                ):
                    raise RuntimeError(
                        f"Unexpected chunk shape: "
                        f"{active_chunk.shape}"
                    )

                chunk_cursor = 0
                chunk_index += 1

                print(
                    f"replan "
                    f"chunk={chunk_index} "
                    f"frame={frame_index} "
                    f"time={data.time:.3f}s"
                )

            # --------------------------------
            # observation_t / action_t
            # --------------------------------

            state_before = gen.get_state(
                model,
                data,
                list(
                    dataset_config[
                        "state"
                    ][
                        "mujoco_joints"
                    ]
                ),
            ).copy()

            raw_action = (
                active_chunk[
                    chunk_cursor
                ]
                .copy()
            )

            chunk_step = chunk_cursor
            chunk_cursor += 1

            if not np.isfinite(
                raw_action
            ).all():
                raise RuntimeError(
                    f"Non-finite action "
                    f"at frame {frame_index}"
                )

            # Preserve raw prediction, but
            # enforce declared actuator range
            # for actual MuJoCo execution.
            applied_action = np.clip(
                raw_action,
                ctrl_ranges[:, 0],
                ctrl_ranges[:, 1],
            )

            clip_delta = np.abs(
                applied_action
                - raw_action
            )

            clipped_count = int(
                np.count_nonzero(
                    clip_delta > 1e-8
                )
            )

            total_clipped_values += (
                clipped_count
            )

            max_clip_delta = max(
                max_clip_delta,
                float(
                    np.max(
                        clip_delta
                    )
                ),
            )

            for actuator, value in zip(
                actuator_ids,
                applied_action,
                strict=True,
            ):
                data.ctrl[
                    actuator
                ] = float(value)

            # --------------------------------
            # Advance MuJoCo to next 30-Hz
            # control timestamp.
            # --------------------------------

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

            if (
                not np.isfinite(
                    data.qpos
                ).all()
                or not np.isfinite(
                    data.qvel
                ).all()
            ):
                raise RuntimeError(
                    "Simulation became "
                    f"non-finite at frame "
                    f"{frame_index}"
                )

            object_z = float(
                data.site_xpos[
                    object_site_id,
                    2,
                ]
            )

            lift_m = (
                object_z
                - initial_object_z
            )

            if lift_m >= minimum_lift_m:
                consecutive_hold += 1
            else:
                consecutive_hold = 0

            maximum_hold = max(
                maximum_hold,
                consecutive_hold,
            )

            if (
                success_frame is None
                and consecutive_hold
                >= hold_frames
            ):
                success_frame = frame_index

            states_before.append(
                state_before.astype(
                    np.float32
                )
            )

            raw_actions.append(
                raw_action.astype(
                    np.float32
                )
            )

            applied_actions.append(
                applied_action.astype(
                    np.float32
                )
            )

            object_z_history.append(
                object_z
            )

            lift_history.append(
                lift_m
            )

            chunk_indices.append(
                chunk_index
            )

            row = {
                "frame":
                    frame_index,
                "time_s":
                    float(data.time),
                "chunk_index":
                    chunk_index,
                "chunk_step":
                    chunk_step,
                "object_z_m":
                    object_z,
                "lift_m":
                    lift_m,
                "lift_mm":
                    lift_m * 1000.0,
                "consecutive_hold_frames":
                    consecutive_hold,
                "clipped_values":
                    clipped_count,
            }

            for idx, name in enumerate(
                base.ACTION_NAMES
            ):
                row[
                    f"state_{name}"
                ] = float(
                    state_before[idx]
                )

                row[
                    f"raw_{name}"
                ] = float(
                    raw_action[idx]
                )

                row[
                    f"applied_{name}"
                ] = float(
                    applied_action[idx]
                )

            rows.append(row)

        # ------------------------------------
        # Result
        # ------------------------------------

        expected_time = (
            args.frames
            / fps
        )

        if not math.isclose(
            data.time,
            expected_time,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                f"Episode ended at "
                f"{data.time:.9f}s; "
                f"expected "
                f"{expected_time:.9f}s"
            )

        success = (
            maximum_hold
            >= hold_frames
        )

        lift_history_np = np.asarray(
            lift_history,
            dtype=np.float32,
        )

        final_lift_m = float(
            lift_history_np[-1]
        )

        max_lift_m = float(
            np.max(
                lift_history_np
            )
        )

        # ------------------------------------
        # Save
        # ------------------------------------

        np.savez_compressed(
            output_dir
            / "rollout.npz",
            states_before=np.asarray(
                states_before,
                dtype=np.float32,
            ),
            raw_actions=np.asarray(
                raw_actions,
                dtype=np.float32,
            ),
            applied_actions=np.asarray(
                applied_actions,
                dtype=np.float32,
            ),
            object_z_m=np.asarray(
                object_z_history,
                dtype=np.float32,
            ),
            lift_m=lift_history_np,
            chunk_index=np.asarray(
                chunk_indices,
                dtype=np.int32,
            ),
        )

        csv_path = (
            output_dir
            / "rollout.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

        metadata = {
            "checkpoint":
                str(base.CHECKPOINT),
            "object_x_m":
                args.object_x,
            "variation_id":
                args.variation_id,
            "noise_seed":
                args.noise_seed,
            "frames":
                args.frames,
            "fps":
                fps,
            "duration_s":
                expected_time,
            "chunk_size":
                int(
                    policy.config.chunk_size
                ),
            "n_action_steps":
                int(
                    policy.config.n_action_steps
                ),
            "execution_horizon":
                int(
                    execution_horizon
                ),
            "replan_rule":
                "replan when current action chunk is exhausted",
            "minimum_lift_delta_m":
                minimum_lift_m,
            "hold_duration_s":
                hold_duration_s,
            "hold_frames":
                hold_frames,
            "success":
                bool(success),
            "success_frame":
                (
                    int(success_frame)
                    if success_frame
                    is not None
                    else None
                ),
            "initial_object_z_m":
                initial_object_z,
            "final_lift_m":
                final_lift_m,
            "max_lift_m":
                max_lift_m,
            "maximum_hold_frames":
                int(maximum_hold),
            "total_clipped_values":
                int(
                    total_clipped_values
                ),
            "max_clip_delta":
                float(max_clip_delta),
        }

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

        print()
        print("=== rollout result ===")
        print(
            "success:",
            success,
        )
        print(
            "final lift:",
            f"{final_lift_m * 1000:.2f} mm",
        )
        print(
            "max lift:",
            f"{max_lift_m * 1000:.2f} mm",
        )
        print(
            "max hold:",
            maximum_hold,
            "frames",
        )
        print(
            "success frame:",
            success_frame,
        )
        print(
            "clipped values:",
            total_clipped_values,
        )
        print(
            "max clip delta:",
            f"{max_clip_delta:.6f}",
        )

        print()
        print("Saved:")
        print(
            output_dir
            / "rollout.npz"
        )
        print(csv_path)
        print(
            output_dir
            / "metadata.json"
        )

        print()
        print(
            "SMOLVLA CLOSED-LOOP "
            "ROLLOUT PASSED"
        )

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
