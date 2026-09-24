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

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts/evaluate_smolvla_action_response.py"
)

DEFAULT_XS = [
    0.220,
    0.230,
    0.240,
    0.250,
    0.260,
]

DEFAULT_SEEDS = [
    20260920,
    20260921,
    20260922,
    20260923,
    20260924,
]

TRAIN_XS = {
    0.220,
    0.230,
    0.240,
    0.250,
    0.260,
}


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
        "--shoulder-lift-offset-deg",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--object-xs",
        type=float,
        nargs="+",
        default=DEFAULT_XS,
    )

    parser.add_argument(
        "--variation-ids",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--noise-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
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

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def render_observation(
    *,
    gen,
    model,
    data,
    renderer,
    dataset_config,
):
    state = gen.get_state(
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


def run_rollout(
    *,
    base,
    gen,
    physics,
    policy,
    preprocessor,
    postprocessor,
    device,
    model,
    data,
    renderer,
    dataset_config,
    task,
    variation,
    object_x,
    shoulder_lift_offset_deg,
    noise_seed,
    frames,
    actuator_ids,
    ctrl_ranges,
    object_site_id,
):
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

    physics_rate_hz = (
        1.0
        / float(
            model.opt.timestep
        )
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Build the original variation-specific initial state.
    # The object remains at the fixed training condition:
    # x = object_x, y = 0.
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
            object_x=float(object_x),
        )
    )

    # home_joint_offsets_deg is ordered consistently with
    # physics.ARM_JOINT_NAMES throughout the frozen dataset
    # generator. Index 1 corresponds to shoulder_lift.
    if len(physics.ARM_JOINT_NAMES) < 2:
        raise RuntimeError(
            "ARM_JOINT_NAMES does not contain shoulder_lift"
        )

    shoulder_lift_joint_name = (
        physics.ARM_JOINT_NAMES[1]
    )

    shoulder_lift_qpos_address = gen.qpos_address(
        model,
        shoulder_lift_joint_name,
    )

    shoulder_lift_actuator = gen.actuator_id(
        model,
        shoulder_lift_joint_name,
    )

    baseline_shoulder_lift_qpos = float(
        data.qpos[shoulder_lift_qpos_address]
    )

    requested_delta_rad = math.radians(
        float(shoulder_lift_offset_deg)
    )

    perturbed_shoulder_lift_qpos = (
        baseline_shoulder_lift_qpos
        + requested_delta_rad
    )

    # Perturb only the initial shoulder_lift state.
    data.qpos[
        shoulder_lift_qpos_address
    ] = perturbed_shoulder_lift_qpos

    data.qvel[:] = 0.0

    # Keep the position actuator initially consistent
    # with the perturbed physical state.
    data.ctrl[
        shoulder_lift_actuator
    ] = perturbed_shoulder_lift_qpos

    mujoco.mj_forward(
        model,
        data,
    )

    actual_shoulder_lift_qpos = float(
        data.qpos[shoulder_lift_qpos_address]
    )

    actual_delta_deg = math.degrees(
        actual_shoulder_lift_qpos
        - baseline_shoulder_lift_qpos
    )

    if not math.isclose(
        actual_delta_deg,
        float(shoulder_lift_offset_deg),
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"shoulder_lift perturbation mismatch: "
            f"requested="
            f"{float(shoulder_lift_offset_deg):.9f} deg, "
            f"actual={actual_delta_deg:.9f} deg"
        )

    # Confirm that the object itself was not perturbed.
    actual_object_x = float(
        data.site_xpos[
            object_site_id,
            0,
        ]
    )

    actual_object_y = float(
        data.site_xpos[
            object_site_id,
            1,
        ]
    )

    if not math.isclose(
        actual_object_x,
        float(object_x),
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Object x changed unexpectedly: "
            f"expected={float(object_x):.9f}, "
            f"actual={actual_object_x:.9f}"
        )

    if not math.isclose(
        actual_object_y,
        float(gen.OBJECT_Y_M),
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Object y changed unexpectedly: "
            f"expected={float(gen.OBJECT_Y_M):.9f}, "
            f"actual={actual_object_y:.9f}"
        )

    # Rebuild observation after perturbing only
    # shoulder_lift.
    joint_names = list(
        dataset_config["state"]["mujoco_joints"]
    )

    state = gen.get_state(
        model,
        data,
        joint_names,
    ).copy()

    observation = {
        "observation.state": state,
    }

    for camera in dataset_config["cameras"].values():
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

    initial_state = (
        observation[
            "observation.state"
        ]
        .copy()
        .astype(np.float32)
    )

    initial_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    # Reset for every rollout.
    # Therefore, a given seed produces the same
    # chunk-noise sequence for all x / variations.
    noise_generator = torch.Generator(
        device=device
    )

    noise_generator.manual_seed(
        int(noise_seed)
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

    active_chunk = None
    chunk_cursor = 0
    chunk_index = -1

    physics_step_count = 0

    consecutive_hold = 0
    maximum_hold = 0
    success_frame = None

    total_clipped_values = 0
    max_clip_delta = 0.0

    clip_count_by_action = np.zeros(
        6,
        dtype=np.int64,
    )

    max_clip_by_action = np.zeros(
        6,
        dtype=np.float64,
    )

    final_lift_m = 0.0
    max_lift_m = -np.inf

    for frame_index in range(frames):
        if (
            active_chunk is None
            or chunk_cursor
            >= execution_horizon
        ):
            if frame_index != 0:
                observation = (
                    render_observation(
                        gen=gen,
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
                    policy.predict_action_chunk(
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

        raw_action = (
            active_chunk[
                chunk_cursor
            ]
            .copy()
        )

        chunk_cursor += 1

        if not np.isfinite(
            raw_action
        ).all():
            raise RuntimeError(
                f"Non-finite action at "
                f"frame={frame_index}"
            )

        applied_action = np.clip(
            raw_action,
            ctrl_ranges[:, 0],
            ctrl_ranges[:, 1],
        )

        clip_delta = np.abs(
            applied_action
            - raw_action
        )

        clipped_mask = (
            clip_delta > 1e-8
        )

        clip_count_by_action += (
            clipped_mask.astype(
                np.int64
            )
        )

        max_clip_by_action = np.maximum(
            max_clip_by_action,
            clip_delta,
        )

        total_clipped_values += int(
            clipped_mask.sum()
        )

        max_clip_delta = max(
            max_clip_delta,
            float(
                clip_delta.max()
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
                f"non-finite at "
                f"frame={frame_index}"
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

        final_lift_m = lift_m
        max_lift_m = max(
            max_lift_m,
            lift_m,
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

    expected_time = (
        frames / fps
    )

    if not math.isclose(
        data.time,
        expected_time,
        abs_tol=1e-8,
    ):
        raise RuntimeError(
            f"Unexpected final time: "
            f"{data.time:.9f}"
        )

    success = (
        maximum_hold
        >= hold_frames
    )

    result = {
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
            float(initial_object_z),
        "final_lift_m":
            float(final_lift_m),
        "max_lift_m":
            float(max_lift_m),
        "maximum_hold_frames":
            int(maximum_hold),
        "total_clipped_values":
            int(total_clipped_values),
        "max_clip_delta":
            float(max_clip_delta),
        "chunk_count":
            int(chunk_index + 1),
        "initial_state":
            initial_state,
        "clip_count_by_action":
            clip_count_by_action,
        "max_clip_by_action":
            max_clip_by_action,
    }

    return result


def load_existing_rows(path):
    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def trial_key(
    seed,
    variation_id,
    object_x,
    shoulder_lift_offset_deg,
):
    return (
        int(seed),
        str(variation_id),
        round(
            float(object_x),
            6,
        ),
        round(
            float(shoulder_lift_offset_deg),
            6,
        ),
    )


def write_aggregates(
    *,
    rows,
    output_dir,
    object_xs,
    shoulder_lift_offset_deg,
    seeds,
):
    parsed = []

    for row in rows:
        parsed.append(
            {
                "noise_seed":
                    int(
                        row[
                            "noise_seed"
                        ]
                    ),
                "variation_id":
                    row[
                        "variation_id"
                    ],
                "object_x_m":
                    float(
                        row[
                            "object_x_m"
                        ]
                    ),
                "shoulder_lift_offset_deg":
                    float(
                        row[
                            "shoulder_lift_offset_deg"
                        ]
                    ),
                "success":
                    int(
                        row[
                            "success"
                        ]
                    ),
                "final_lift_m":
                    float(
                        row[
                            "final_lift_m"
                        ]
                    ),
                "max_lift_m":
                    float(
                        row[
                            "max_lift_m"
                        ]
                    ),
                "total_clipped_values":
                    int(
                        row[
                            "total_clipped_values"
                        ]
                    ),
                "max_clip_delta":
                    float(
                        row[
                            "max_clip_delta"
                        ]
                    ),
            }
        )

    position_rows = []

    for object_x in object_xs:
        subset = [
            row
            for row in parsed
            if math.isclose(
                row["object_x_m"],
                object_x,
                abs_tol=1e-9,
            )
        ]

        if not subset:
            continue

        successes = sum(
            row["success"]
            for row in subset
        )

        position_type = (
            "train_anchor"
            if any(
                math.isclose(
                    object_x,
                    train_x,
                    abs_tol=1e-9,
                )
                for train_x
                in TRAIN_XS
            )
            else "unseen_interpolation"
        )

        position_rows.append(
            {
                "object_x_m":
                    object_x,
                "shoulder_lift_offset_deg":
                    float(shoulder_lift_offset_deg),
                "position_type":
                    position_type,
                "shoulder_lift_position_type":
                    (
                        "baseline_lift"
                        if abs(float(shoulder_lift_offset_deg)) < 1e-12
                        else "unseen_lift"
                    ),
                "n":
                    len(subset),
                "successes":
                    successes,
                "success_rate":
                    successes
                    / len(subset),
                "final_lift_mm_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "final_lift_m"
                                ]
                                * 1000.0
                                for row
                                in subset
                            ]
                        )
                    ),
                "max_lift_mm_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "max_lift_m"
                                ]
                                * 1000.0
                                for row
                                in subset
                            ]
                        )
                    ),
                "clip_values_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "total_clipped_values"
                                ]
                                for row
                                in subset
                            ]
                        )
                    ),
                "max_clip_delta":
                    float(
                        max(
                            row[
                                "max_clip_delta"
                            ]
                            for row
                            in subset
                        )
                    ),
            }
        )

    position_path = (
        output_dir
        / "position_summary.csv"
    )

    if position_rows:
        with position_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    position_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                position_rows
            )

    seed_position_rows = []

    for seed in seeds:
        for object_x in object_xs:
            subset = [
                row
                for row in parsed
                if (
                    row["noise_seed"]
                    == seed
                    and math.isclose(
                        row["object_x_m"],
                        object_x,
                        abs_tol=1e-9,
                    )
                )
            ]

            if not subset:
                continue

            successes = sum(
                row["success"]
                for row in subset
            )

            seed_position_rows.append(
                {
                    "noise_seed":
                        seed,
                    "object_x_m":
                        object_x,
                    "n":
                        len(subset),
                    "successes":
                        successes,
                    "success_rate":
                        successes
                        / len(subset),
                }
            )

    seed_position_path = (
        output_dir
        / "seed_position_summary.csv"
    )

    if seed_position_rows:
        with seed_position_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    seed_position_rows[
                        0
                    ].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                seed_position_rows
            )

    return position_rows


def main():
    args = parse_args()

    checkpoint = args.checkpoint.resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint}"
        )

    output_dir = (
        args.output_dir.resolve()
    )

    trial_csv = (
        output_dir
        / "trial_summary.csv"
    )

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(
                output_dir
            )
        elif not args.resume:
            raise SystemExit(
                f"Output exists:\n"
                f"{output_dir}\n"
                f"Use --resume or "
                f"--overwrite."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows = (
        load_existing_rows(
            trial_csv
        )
        if args.resume
        else []
    )

    completed = {
        trial_key(
            row["noise_seed"],
            row["variation_id"],
            row["object_x_m"],
            row.get(
                "shoulder_lift_offset_deg",
                args.shoulder_lift_offset_deg,
            ),
        )
        for row in existing_rows
    }

    base = load_base_module()

    gen = base.load_generator_module()
    physics = gen.load_physics_module()

    dataset_config = base.load_yaml(
        base.DATASET_CONFIG_PATH
    )

    task = base.load_yaml(
        base.TASK_CONFIG_PATH
    )["task"]

    all_variations = (
        gen.load_variations()
    )

    if args.variation_ids is None:
        variations = all_variations
    else:
        requested = set(
            args.variation_ids
        )

        variations = [
            item
            for item in all_variations
            if item["id"]
            in requested
        ]

        found = {
            item["id"]
            for item in variations
        }

        missing = (
            requested - found
        )

        if missing:
            raise RuntimeError(
                f"Unknown variation IDs: "
                f"{sorted(missing)}"
            )

    object_xs = [
        float(x)
        for x in args.object_xs
    ]

    seeds = [
        int(seed)
        for seed
        in args.noise_seeds
    ]

    total_requested = (
        len(seeds)
        * len(variations)
        * len(object_xs)
    )

    print("=== batch condition ===")
    print(
        "positions:",
        object_xs,
    )
    print(
        "variations:",
        len(variations),
    )
    print(
        "seeds:",
        seeds,
    )
    print(
        "requested trials:",
        total_requested,
    )
    print(
        "already completed:",
        len(completed),
    )

    device = torch.device(
        "cuda"
    )

    print()
    print("=== load policy once ===")

    policy = (
        PI05Policy
        .from_pretrained(
            str(
                checkpoint
            )
        )
    )

    policy = policy.to(device)
    policy.eval()

    (
        preprocessor,
        postprocessor,
    ) = base.make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(
            checkpoint
        ),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
        },
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            gen.SCENE_PATH
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

    arm_actuators = [
        gen.actuator_id(
            model,
            name,
        )
        for name
        in physics.ARM_JOINT_NAMES
    ]

    gripper_actuator = (
        gen.actuator_id(
            model,
            physics.GRIPPER_JOINT_NAME,
        )
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
            "target_box_center "
            "site not found"
        )

    fieldnames = [
        "noise_seed",
        "variation_id",
        "object_x_m",
        "shoulder_lift_offset_deg",
        "position_type",
        "shoulder_lift_position_type",
        "success",
        "success_frame",
        "final_lift_m",
        "max_lift_m",
        "maximum_hold_frames",
        "total_clipped_values",
        "max_clip_delta",
        "chunk_count",
        "initial_state_0",
        "initial_state_1",
        "initial_state_2",
        "initial_state_3",
        "initial_state_4",
        "initial_state_5",
    ]

    for name in base.ACTION_NAMES:
        fieldnames.append(
            f"clip_count_{name}"
        )
        fieldnames.append(
            f"max_clip_{name}"
        )

    new_rows = []

    # Used to verify same initial robot state
    # across x for each fixed variation.
    initial_state_refs = {}

    run_counter = 0

    try:
        with trial_csv.open(
            "a"
            if args.resume
            and trial_csv.exists()
            else "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if (
                not args.resume
                or trial_csv.stat().st_size
                == 0
            ):
                writer.writeheader()

            for seed in seeds:
                for variation in variations:
                    variation_id = (
                        variation["id"]
                    )

                    for object_x in object_xs:
                        key = trial_key(
                            seed,
                            variation_id,
                            object_x,
                            args.shoulder_lift_offset_deg,
                        )

                        if key in completed:
                            continue

                        run_counter += 1

                        result = run_rollout(
                            base=base,
                            gen=gen,
                            physics=physics,
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            device=device,
                            model=model,
                            data=data,
                            renderer=renderer,
                            dataset_config=dataset_config,
                            task=task,
                            variation=variation,
                            object_x=object_x,
                            shoulder_lift_offset_deg=args.shoulder_lift_offset_deg,
                            noise_seed=seed,
                            frames=args.frames,
                            actuator_ids=actuator_ids,
                            ctrl_ranges=ctrl_ranges,
                            object_site_id=object_site_id,
                        )

                        ref_key = (
                            variation_id
                        )

                        initial_state = (
                            result[
                                "initial_state"
                            ]
                        )

                        if (
                            ref_key
                            not in
                            initial_state_refs
                        ):
                            initial_state_refs[
                                ref_key
                            ] = (
                                initial_state
                                .copy()
                            )
                        else:
                            diff = float(
                                np.max(
                                    np.abs(
                                        initial_state
                                        - initial_state_refs[
                                            ref_key
                                        ]
                                    )
                                )
                            )

                            if diff > 1e-6:
                                raise RuntimeError(
                                    f"Initial state "
                                    f"changed across x: "
                                    f"{variation_id} "
                                    f"diff={diff}"
                                )

                        position_type = (
                            "train_anchor"
                            if any(
                                math.isclose(
                                    object_x,
                                    train_x,
                                    abs_tol=1e-9,
                                )
                                for train_x
                                in TRAIN_XS
                            )
                            else
                            "unseen_interpolation"
                        )

                        row = {
                            "noise_seed":
                                seed,
                            "variation_id":
                                variation_id,
                            "object_x_m":
                                object_x,
                            "shoulder_lift_offset_deg":
                                float(
                                    args.shoulder_lift_offset_deg
                                ),
                            "position_type":
                                position_type,
                            "shoulder_lift_position_type":
                                (
                                    "baseline_lift"
                                    if abs(
                                        float(
                                            args.shoulder_lift_offset_deg
                                        )
                                    ) < 1e-12
                                    else "unseen_lift"
                                ),
                            "success":
                                int(
                                    result[
                                        "success"
                                    ]
                                ),
                            "success_frame":
                                (
                                    result[
                                        "success_frame"
                                    ]
                                    if result[
                                        "success_frame"
                                    ]
                                    is not None
                                    else ""
                                ),
                            "final_lift_m":
                                result[
                                    "final_lift_m"
                                ],
                            "max_lift_m":
                                result[
                                    "max_lift_m"
                                ],
                            "maximum_hold_frames":
                                result[
                                    "maximum_hold_frames"
                                ],
                            "total_clipped_values":
                                result[
                                    "total_clipped_values"
                                ],
                            "max_clip_delta":
                                result[
                                    "max_clip_delta"
                                ],
                            "chunk_count":
                                result[
                                    "chunk_count"
                                ],
                        }

                        for i in range(6):
                            row[
                                f"initial_state_{i}"
                            ] = float(
                                initial_state[i]
                            )

                        for i, name in enumerate(
                            base.ACTION_NAMES
                        ):
                            row[
                                f"clip_count_{name}"
                            ] = int(
                                result[
                                    "clip_count_by_action"
                                ][i]
                            )

                            row[
                                f"max_clip_{name}"
                            ] = float(
                                result[
                                    "max_clip_by_action"
                                ][i]
                            )

                        writer.writerow(
                            row
                        )
                        f.flush()

                        new_rows.append(
                            {
                                key: str(value)
                                for key, value
                                in row.items()
                            }
                        )

                        completed.add(
                            key
                        )

                        print(
                            f"[{run_counter:04d}] "
                            f"seed={seed} "
                            f"{variation_id} "
                            f"x={object_x:.3f} "
                            f"success="
                            f"{int(result['success'])} "
                            f"max_lift="
                            f"{result['max_lift_m'] * 1000:.1f}mm "
                            f"clips="
                            f"{result['total_clipped_values']}"
                        )

    finally:
        renderer.close()

    all_rows = (
        existing_rows
        + new_rows
    )

    position_rows = write_aggregates(
        rows=all_rows,
        output_dir=output_dir,
        object_xs=object_xs,
        shoulder_lift_offset_deg=args.shoulder_lift_offset_deg,
        seeds=seeds,
    )

    metadata = {
        "checkpoint":
            str(
                checkpoint
            ),
        "object_x_m":
            object_xs,
        "shoulder_lift_offset_deg":
            float(args.shoulder_lift_offset_deg),
        "training_object_y_m":
            [
                0.0,
            ],
        "shoulder_lift_position_type":
            (
                "baseline_lift"
                if abs(
                    float(args.shoulder_lift_offset_deg)
                ) < 1e-12
                else "unseen_lift"
            ),
        "position_type_semantics":
            (
                "position_type classifies object_x; "
                "shoulder_lift_position_type classifies shoulder_lift_offset_deg."
            ),
        "training_anchor_x_m":
            sorted(
                TRAIN_XS
            ),
        "variation_ids":
            [
                variation["id"]
                for variation
                in variations
            ],
        "noise_seeds":
            seeds,
        "noise_pairing":
            (
                "For each noise seed, the RNG is "
                "reset at the start of every rollout, "
                "so all positions and variations receive "
                "the same sequence of chunk noises."
            ),
        "frames":
            args.frames,
        "fps":
            float(
                task[
                    "phases"
                ][
                    "timing"
                ][
                    "rate_hz"
                ]
            ),
        "success_minimum_lift_m":
            float(
                task[
                    "success"
                ][
                    "minimum_lift_delta_m"
                ]
            ),
        "success_hold_duration_s":
            float(
                task[
                    "success"
                ][
                    "hold_duration_s"
                ]
            ),
        "execution_horizon":
            int(
                min(
                    policy.config.n_action_steps,
                    policy.config.chunk_size,
                )
            ),
        "script_or_ik_during_rollout":
            False,
        "action_execution":
            (
                "PI05 postprocessed action; "
                "clipped only to MuJoCo actuator ctrlrange."
            ),
        "completed_trials":
            len(all_rows),
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
    print("=== position summary ===")

    for row in position_rows:
        print(
            f"x={row['object_x_m']:.3f} "
            f"dlift={row['shoulder_lift_offset_deg']:+.1f}deg "
            f"{row['position_type']:20s} "
            f"{row['shoulder_lift_position_type']:10s} "
            f"{row['successes']:3d}/"
            f"{row['n']:3d} "
            f"rate="
            f"{row['success_rate']:.3f} "
            f"max_lift_mean="
            f"{row['max_lift_mm_mean']:.1f}mm"
        )

    print()
    print(
        "completed trials:",
        len(all_rows),
    )
    print(
        "trial summary:",
        trial_csv,
    )
    print(
        "position summary:",
        output_dir
        / "position_summary.csv",
    )
    print()
    print(
        "PI05 TEACHER CLOSED-LOOP "
        "BATCH EVALUATION PASSED"
    )


if __name__ == "__main__":
    main()
