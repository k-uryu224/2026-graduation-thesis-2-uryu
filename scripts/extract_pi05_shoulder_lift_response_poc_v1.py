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
import pyarrow.parquet as pq
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts/evaluate_smolvla_action_response.py"
)

DEFAULT_PROTOCOL = (
    ROOT
    / "configs/evaluation/"
    "lprd_shoulder_lift_poc_v1_protocol.json"
)

DEFAULT_IMPLEMENTATION_CONFIG = (
    ROOT
    / "configs/evaluation/"
    "lprd_shoulder_lift_poc_v1_teacher_response_implementation.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/lprd_shoulder_lift_poc_v1/"
    "teacher_response"
)

ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )

    parser.add_argument(
        "--implementation-config",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_CONFIG,
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
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_commit():
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "smolvla_action_response_base",
        BASE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {BASE_SCRIPT}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def offset_key(delta_deg: float):
    value = int(
        round(abs(float(delta_deg)))
    )

    if delta_deg < 0:
        prefix = "m"
    else:
        prefix = "p"

    return f"{prefix}{value:03d}"


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
):
    x = np.asarray(
        a,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        b,
        dtype=np.float64,
    ).reshape(-1)

    nx = float(
        np.linalg.norm(x)
    )

    ny = float(
        np.linalg.norm(y)
    )

    if (
        nx < 1e-12
        or ny < 1e-12
    ):
        return float("nan")

    return float(
        np.dot(x, y)
        / (nx * ny)
    )


def rms(x: np.ndarray):
    x64 = np.asarray(
        x,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(x64)
            )
        )
    )


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
            camera=camera[
                "mujoco_name"
            ],
        )

        observation[
            camera["lerobot_key"]
        ] = (
            renderer.render()
            .copy()
        )

    return observation


def build_observation_at_offset(
    *,
    base,
    gen,
    physics,
    model,
    data,
    renderer,
    dataset_config,
    task,
    variation,
    object_x,
    shoulder_lift_offset_deg,
    object_site_id,
):
    # Rebuild the frozen variation-specific reference state
    # before applying each perturbation.
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

    if len(physics.ARM_JOINT_NAMES) < 2:
        raise RuntimeError(
            "ARM_JOINT_NAMES does not contain shoulder_lift"
        )

    joint_name = (
        physics.ARM_JOINT_NAMES[1]
    )

    qpos_address = gen.qpos_address(
        model,
        joint_name,
    )

    actuator = gen.actuator_id(
        model,
        joint_name,
    )

    baseline_qpos = float(
        data.qpos[qpos_address]
    )

    requested_delta_rad = (
        math.radians(
            float(
                shoulder_lift_offset_deg
            )
        )
    )

    perturbed_qpos = (
        baseline_qpos
        + requested_delta_rad
    )

    data.qpos[
        qpos_address
    ] = perturbed_qpos

    data.qvel[:] = 0.0

    data.ctrl[
        actuator
    ] = perturbed_qpos

    mujoco.mj_forward(
        model,
        data,
    )

    actual_qpos = float(
        data.qpos[qpos_address]
    )

    actual_delta_deg = (
        math.degrees(
            actual_qpos
            - baseline_qpos
        )
    )

    if not math.isclose(
        actual_delta_deg,
        float(
            shoulder_lift_offset_deg
        ),
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "shoulder_lift perturbation mismatch: "
            f"requested="
            f"{shoulder_lift_offset_deg:.9f}, "
            f"actual="
            f"{actual_delta_deg:.9f}"
        )

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
            "Object x changed unexpectedly"
        )

    if not math.isclose(
        actual_object_y,
        float(gen.OBJECT_Y_M),
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Object y changed unexpectedly"
        )

    observation = (
        render_observation(
            gen=gen,
            model=model,
            data=data,
            renderer=renderer,
            dataset_config=dataset_config,
        )
    )

    return observation


def load_demo_actions(
    *,
    dataset_root,
    object_x,
    variation_ids,
    expected_episode_indices,
):
    manifest_path = (
        dataset_root
        / "experiment_manifest.csv"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = list(
            csv.DictReader(f)
        )

    mapping = {}

    for variation_id in variation_ids:
        matches = [
            row
            for row in manifest
            if (
                abs(
                    float(
                        row[
                            "object_x_m"
                        ]
                    )
                    - float(object_x)
                )
                < 1e-12
                and row[
                    "variation_id"
                ]
                == variation_id
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one manifest row for "
                f"x={object_x}, "
                f"{variation_id}; "
                f"got {len(matches)}"
            )

        episode_index = int(
            matches[0][
                "episode_index"
            ]
        )

        expected = int(
            expected_episode_indices[
                variation_id
            ]
        )

        if episode_index != expected:
            raise RuntimeError(
                f"{variation_id}: "
                f"expected episode "
                f"{expected}, "
                f"got {episode_index}"
            )

        mapping[
            variation_id
        ] = episode_index

    parquet_files = sorted(
        dataset_root.glob(
            "data/chunk-*/file-*.parquet"
        )
    )

    if not parquet_files:
        raise RuntimeError(
            "Dataset parquet not found"
        )

    tables = [
        pq.read_table(path)
        for path in parquet_files
    ]

    if len(tables) == 1:
        table = tables[0]
    else:
        import pyarrow as pa

        table = pa.concat_tables(
            tables
        )

    episode_column = np.asarray(
        table[
            "episode_index"
        ].to_numpy()
    ).astype(np.int64)

    frame_column = np.asarray(
        table[
            "frame_index"
        ].to_numpy()
    ).astype(np.int64)

    action_array = np.asarray(
        table[
            "action"
        ].to_pylist(),
        dtype=np.float32,
    )

    if (
        action_array.ndim != 2
        or action_array.shape[1] != 6
    ):
        raise RuntimeError(
            f"Unexpected action shape: "
            f"{action_array.shape}"
        )

    demo_chunks = []

    for variation_id in variation_ids:
        episode_index = mapping[
            variation_id
        ]

        indexes = np.flatnonzero(
            episode_column
            == episode_index
        )

        if len(indexes) != 210:
            raise RuntimeError(
                f"{variation_id}: "
                f"expected 210 frames, "
                f"got {len(indexes)}"
            )

        indexes = indexes[
            np.argsort(
                frame_column[indexes]
            )
        ]

        frames = frame_column[
            indexes
        ]

        if not np.array_equal(
            frames,
            np.arange(
                210,
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                f"{variation_id}: "
                "unexpected frame_index sequence"
            )

        chunk = (
            action_array[
                indexes[:50]
            ]
            .copy()
            .astype(np.float32)
        )

        if chunk.shape != (
            50,
            6,
        ):
            raise RuntimeError(
                f"{variation_id}: "
                f"unexpected chunk shape "
                f"{chunk.shape}"
            )

        demo_chunks.append(
            chunk
        )

    return (
        mapping,
        np.stack(
            demo_chunks,
            axis=0,
        ),
        manifest_path,
        parquet_files,
    )


def predict_teacher_chunk(
    *,
    base,
    policy,
    preprocessor,
    postprocessor,
    device,
    observation,
    noise,
):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    batch = (
        base.process_observation(
            observation=observation,
            preprocessor=preprocessor,
            device=device,
        )
    )

    with torch.inference_mode():
        raw_chunk = (
            policy.predict_action_chunk(
                batch,
                noise=noise.clone(),
            )
        )

        processed_chunk = (
            postprocessor(
                raw_chunk.clone()
            )
        )

    action = (
        processed_chunk[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if action.shape != (
        policy.config.chunk_size,
        6,
    ):
        raise RuntimeError(
            f"Unexpected teacher action "
            f"shape: {action.shape}"
        )

    return action


def main():
    args = parse_args()

    protocol_path = (
        args.protocol.resolve()
    )

    implementation_path = (
        args.implementation_config
        .resolve()
    )

    protocol = load_json(
        protocol_path
    )

    implementation = load_json(
        implementation_path
    )

    variation_ids = list(
        protocol[
            "distillation_split"
        ][
            "variation_ids"
        ]
    )

    response_seeds = [
        int(seed)
        for seed in protocol[
            "response"
        ][
            "teacher_response_noise_seeds"
        ]
    ]

    perturb_offsets = [
        float(delta)
        for delta in protocol[
            "perturbation"
        ][
            "distillation_offsets_deg"
        ]
    ]

    reference_offset = float(
        protocol[
            "perturbation"
        ][
            "reference_offset_deg"
        ]
    )

    heldout_offset = float(
        protocol[
            "perturbation"
        ][
            "primary_heldout_evaluation_offset_deg"
        ]
    )

    if any(
        math.isclose(
            delta,
            heldout_offset,
            abs_tol=1e-12,
        )
        for delta in perturb_offsets
    ):
        raise RuntimeError(
            "Held-out -5deg condition must "
            "not appear in distillation offsets"
        )

    task_condition = (
        implementation[
            "task_condition"
        ]
    )

    object_x = float(
        task_condition[
            "object_x_m"
        ]
    )

    dataset_root = (
        ROOT
        / implementation[
            "dataset_root"
        ]
    ).resolve()

    expected_episode_indices = (
        implementation[
            "reference_demo"
        ][
            "expected_episode_indices"
        ]
    )

    (
        episode_mapping,
        demo_action_ref,
        manifest_path,
        parquet_files,
    ) = load_demo_actions(
        dataset_root=dataset_root,
        object_x=object_x,
        variation_ids=variation_ids,
        expected_episode_indices=(
            expected_episode_indices
        ),
    )

    print(
        "=== teacher response input validation ==="
    )

    print(
        "object_x:",
        object_x,
    )

    print(
        "variations:",
        variation_ids,
    )

    print(
        "episodes:",
        episode_mapping,
    )

    print(
        "demo_action_ref:",
        demo_action_ref.shape,
    )

    print(
        "response seeds:",
        response_seeds,
    )

    print(
        "reference offset:",
        reference_offset,
    )

    print(
        "perturb offsets:",
        perturb_offsets,
    )

    print(
        "held-out offset:",
        heldout_offset,
    )

    if demo_action_ref.shape != (
        len(variation_ids),
        50,
        6,
    ):
        raise RuntimeError(
            "Unexpected demo action shape"
        )

    if len(response_seeds) != 8:
        raise RuntimeError(
            "Expected exactly 8 teacher "
            "response noise seeds"
        )

    if sorted(
        perturb_offsets
    ) != [
        -6.0,
        -4.0,
    ]:
        raise RuntimeError(
            "Expected distillation offsets "
            "[-6, -4] deg"
        )

    if not math.isclose(
        reference_offset,
        0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Expected reference offset 0deg"
        )

    if not math.isclose(
        heldout_offset,
        -5.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Expected held-out offset -5deg"
        )

    if args.validate_only:
        print(
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
                f"Output already exists:\n"
                f"{output_dir}\n"
                "Use --overwrite only if "
                "intentional."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = load_base_module()

    gen = base.load_generator_module()
    physics = gen.load_physics_module()

    dataset_config = base.load_yaml(
        base.DATASET_CONFIG_PATH
    )

    task = base.load_yaml(
        base.TASK_CONFIG_PATH
    )["task"]

    all_variations = {
        item["id"]: item
        for item
        in gen.load_variations()
    }

    variations = []

    for variation_id in variation_ids:
        if variation_id not in all_variations:
            raise RuntimeError(
                f"Unknown variation: "
                f"{variation_id}"
            )

        variations.append(
            all_variations[
                variation_id
            ]
        )

    checkpoint = (
        ROOT
        / protocol[
            "teacher"
        ][
            "checkpoint"
        ]
    ).resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            checkpoint
        )

    device = torch.device(
        "cuda"
    )

    print()
    print("=== load pi0.5 teacher ===")
    print(checkpoint)

    policy = (
        PI05Policy.from_pretrained(
            str(checkpoint)
        )
    )

    policy = policy.to(device)
    policy.eval()

    if (
        int(policy.config.chunk_size)
        != 50
    ):
        raise RuntimeError(
            f"Expected chunk_size=50, "
            f"got "
            f"{policy.config.chunk_size}"
        )

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
        str(gen.SCENE_PATH)
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
        dtype=np.int64,
    )

    ctrl_ranges = (
        model.actuator_ctrlrange[
            actuator_ids
        ]
        .copy()
        .astype(np.float32)
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
            "target_box_center site not found"
        )

    n_variations = len(
        variation_ids
    )

    n_seeds = len(
        response_seeds
    )

    teacher_action_ref = np.empty(
        (
            n_variations,
            n_seeds,
            50,
            6,
        ),
        dtype=np.float32,
    )

    teacher_actions = {
        delta: np.empty(
            (
                n_variations,
                n_seeds,
                50,
                6,
            ),
            dtype=np.float32,
        )
        for delta
        in perturb_offsets
    }

    by_noise_rows = []

    noise_shape = (
        1,
        int(
            policy.config.chunk_size
        ),
        int(
            policy.config.max_action_dim
        ),
    )

    try:
        for variation_index, variation in enumerate(
            variations
        ):
            variation_id = (
                variation["id"]
            )

            print()
            print(
                "=== variation "
                f"{variation_id} ==="
            )

            observations = {}

            for delta in [
                reference_offset,
                *perturb_offsets,
            ]:
                observations[
                    delta
                ] = (
                    build_observation_at_offset(
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

            for seed_index, seed in enumerate(
                response_seeds
            ):
                generator = torch.Generator(
                    device=device
                )

                generator.manual_seed(
                    int(seed)
                )

                # Exactly one noise tensor is generated
                # for this seed and reused for 0, -4, -6.
                common_noise = torch.randn(
                    noise_shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )

                ref_action = (
                    predict_teacher_chunk(
                        base=base,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        observation=observations[
                            reference_offset
                        ],
                        noise=common_noise,
                    )
                )

                teacher_action_ref[
                    variation_index,
                    seed_index,
                ] = ref_action

                for delta in perturb_offsets:
                    perturbed_action = (
                        predict_teacher_chunk(
                            base=base,
                            policy=policy,
                            preprocessor=(
                                preprocessor
                            ),
                            postprocessor=(
                                postprocessor
                            ),
                            device=device,
                            observation=(
                                observations[
                                    delta
                                ]
                            ),
                            noise=common_noise,
                        )
                    )

                    teacher_actions[
                        delta
                    ][
                        variation_index,
                        seed_index,
                    ] = perturbed_action

                    response = (
                        perturbed_action
                        - ref_action
                    )

                    row = {
                        "variation_id":
                            variation_id,
                        "episode_index":
                            episode_mapping[
                                variation_id
                            ],
                        "noise_seed":
                            seed,
                        "delta_deg":
                            delta,
                        "response_rms":
                            rms(response),
                    }

                    for dim, name in enumerate(
                        ACTION_NAMES
                    ):
                        row[
                            f"{name}_response_rms"
                        ] = rms(
                            response[
                                :,
                                dim,
                            ]
                        )

                    by_noise_rows.append(
                        row
                    )

                print(
                    f"seed {seed}: done"
                )

    finally:
        renderer.close()

    responses = {
        delta: (
            teacher_actions[delta]
            - teacher_action_ref
        )
        for delta
        in perturb_offsets
    }

    mean_responses = {
        delta: np.mean(
            responses[delta],
            axis=1,
        ).astype(np.float32)
        for delta
        in perturb_offsets
    }

    std_responses = {
        delta: np.std(
            responses[delta],
            axis=1,
        ).astype(np.float32)
        for delta
        in perturb_offsets
    }

    mean_teacher_actions = {
        delta: np.mean(
            teacher_actions[delta],
            axis=1,
        ).astype(np.float32)
        for delta
        in perturb_offsets
    }

    teacher_action_ref_mean = (
        np.mean(
            teacher_action_ref,
            axis=1,
        )
        .astype(np.float32)
    )

    response_targets_preclip = {}
    response_targets = {}
    clip_masks = {}

    ctrl_low = (
        ctrl_ranges[:, 0]
        .reshape(1, 1, 6)
    )

    ctrl_high = (
        ctrl_ranges[:, 1]
        .reshape(1, 1, 6)
    )

    for delta in perturb_offsets:
        preclip = (
            demo_action_ref
            + mean_responses[
                delta
            ]
        ).astype(np.float32)

        clipped = np.clip(
            preclip,
            ctrl_low,
            ctrl_high,
        ).astype(np.float32)

        mask = (
            np.abs(
                preclip
                - clipped
            )
            > 1e-9
        )

        response_targets_preclip[
            delta
        ] = preclip

        response_targets[
            delta
        ] = clipped

        clip_masks[
            delta
        ] = mask

    summary_rows = []

    for variation_index, variation_id in enumerate(
        variation_ids
    ):
        for delta in perturb_offsets:
            response_seed = (
                responses[
                    delta
                ][
                    variation_index
                ]
            )

            mean_response = (
                mean_responses[
                    delta
                ][
                    variation_index
                ]
            )

            deviations = (
                response_seed
                - mean_response[
                    None,
                    :,
                    :,
                ]
            )

            cosines = [
                cosine_similarity(
                    response_seed[
                        seed_index
                    ],
                    mean_response,
                )
                for seed_index
                in range(n_seeds)
            ]

            clip_mask = (
                clip_masks[
                    delta
                ][
                    variation_index
                ]
            )

            preclip = (
                response_targets_preclip[
                    delta
                ][
                    variation_index
                ]
            )

            clipped = (
                response_targets[
                    delta
                ][
                    variation_index
                ]
            )

            row = {
                "variation_id":
                    variation_id,
                "episode_index":
                    episode_mapping[
                        variation_id
                    ],
                "delta_deg":
                    delta,
                "mean_response_rms":
                    rms(
                        mean_response
                    ),
                "seed_dispersion_rms":
                    rms(
                        deviations
                    ),
                "mean_seed_cosine_to_mean":
                    float(
                        np.nanmean(
                            cosines
                        )
                    ),
                "target_clip_count":
                    int(
                        np.count_nonzero(
                            clip_mask
                        )
                    ),
                "target_clip_fraction":
                    float(
                        np.mean(
                            clip_mask
                        )
                    ),
                "target_max_clip_delta":
                    float(
                        np.max(
                            np.abs(
                                preclip
                                - clipped
                            )
                        )
                    ),
            }

            for dim, name in enumerate(
                ACTION_NAMES
            ):
                row[
                    f"{name}_mean_response_rms"
                ] = rms(
                    mean_response[
                        :,
                        dim,
                    ]
                )

            summary_rows.append(
                row
            )

    cross_rows = []

    if len(perturb_offsets) == 2:
        delta_a = perturb_offsets[0]
        delta_b = perturb_offsets[1]

        for variation_index, variation_id in enumerate(
            variation_ids
        ):
            a = mean_responses[
                delta_a
            ][
                variation_index
            ]

            b = mean_responses[
                delta_b
            ][
                variation_index
            ]

            norm_a = float(
                np.linalg.norm(
                    a.reshape(-1)
                )
            )

            norm_b = float(
                np.linalg.norm(
                    b.reshape(-1)
                )
            )

            cross_rows.append(
                {
                    "variation_id":
                        variation_id,
                    "delta_a_deg":
                        delta_a,
                    "delta_b_deg":
                        delta_b,
                    "cosine_similarity":
                        cosine_similarity(
                            a,
                            b,
                        ),
                    "norm_a":
                        norm_a,
                    "norm_b":
                        norm_b,
                    "norm_b_over_a":
                        (
                            norm_b
                            / norm_a
                            if norm_a
                            > 1e-12
                            else float("nan")
                        ),
                }
            )

        all_a = (
            mean_responses[
                delta_a
            ]
            .reshape(-1)
        )

        all_b = (
            mean_responses[
                delta_b
            ]
            .reshape(-1)
        )

        norm_a = float(
            np.linalg.norm(
                all_a
            )
        )

        norm_b = float(
            np.linalg.norm(
                all_b
            )
        )

        cross_rows.append(
            {
                "variation_id":
                    "ALL",
                "delta_a_deg":
                    delta_a,
                "delta_b_deg":
                    delta_b,
                "cosine_similarity":
                    cosine_similarity(
                        all_a,
                        all_b,
                    ),
                "norm_a":
                    norm_a,
                "norm_b":
                    norm_b,
                "norm_b_over_a":
                    (
                        norm_b
                        / norm_a
                        if norm_a
                        > 1e-12
                        else float("nan")
                    ),
            }
        )

    npz_payload = {
        "variation_ids":
            np.asarray(
                variation_ids
            ),
        "episode_indices":
            np.asarray(
                [
                    episode_mapping[
                        variation_id
                    ]
                    for variation_id
                    in variation_ids
                ],
                dtype=np.int64,
            ),
        "teacher_response_noise_seeds":
            np.asarray(
                response_seeds,
                dtype=np.int64,
            ),
        "demo_action_ref":
            demo_action_ref,
        "teacher_action_ref":
            teacher_action_ref,
        "teacher_action_ref_mean":
            teacher_action_ref_mean,
        "actuator_ctrl_ranges":
            ctrl_ranges,
    }

    for delta in perturb_offsets:
        key = offset_key(
            delta
        )

        npz_payload[
            f"teacher_action_{key}"
        ] = teacher_actions[
            delta
        ]

        npz_payload[
            f"teacher_action_{key}_mean"
        ] = mean_teacher_actions[
            delta
        ]

        npz_payload[
            f"teacher_response_{key}"
        ] = responses[
            delta
        ]

        npz_payload[
            f"mean_response_{key}"
        ] = mean_responses[
            delta
        ]

        npz_payload[
            f"std_response_{key}"
        ] = std_responses[
            delta
        ]

        npz_payload[
            f"response_target_preclip_{key}"
        ] = response_targets_preclip[
            delta
        ]

        npz_payload[
            f"response_target_{key}"
        ] = response_targets[
            delta
        ]

        npz_payload[
            f"response_target_clip_mask_{key}"
        ] = clip_masks[
            delta
        ]

    np.savez_compressed(
        output_dir
        / "responses.npz",
        **npz_payload,
    )

    def write_csv(
        path,
        rows,
    ):
        if not rows:
            return

        with path.open(
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
            writer.writerows(
                rows
            )

    write_csv(
        output_dir
        / "response_by_noise.csv",
        by_noise_rows,
    )

    write_csv(
        output_dir
        / "response_summary.csv",
        summary_rows,
    )

    write_csv(
        output_dir
        / "response_cross_delta.csv",
        cross_rows,
    )

    metadata = {
        "experiment_id":
            protocol[
                "experiment_id"
            ],
        "source_git_commit":
            git_commit(),
        "protocol":
            str(
                protocol_path
            ),
        "protocol_sha256":
            sha256_file(
                protocol_path
            ),
        "implementation_config":
            str(
                implementation_path
            ),
        "implementation_config_sha256":
            sha256_file(
                implementation_path
            ),
        "teacher_checkpoint":
            str(
                checkpoint
            ),
        "dataset_root":
            str(
                dataset_root
            ),
        "manifest_sha256":
            sha256_file(
                manifest_path
            ),
        "parquet_files": [
            {
                "path": str(path),
                "sha256":
                    sha256_file(
                        path
                    ),
            }
            for path
            in parquet_files
        ],
        "object_x_m":
            object_x,
        "variation_ids":
            variation_ids,
        "episode_mapping":
            episode_mapping,
        "reference_offset_deg":
            reference_offset,
        "perturb_offsets_deg":
            perturb_offsets,
        "heldout_offset_deg":
            heldout_offset,
        "teacher_response_noise_seeds":
            response_seeds,
        "chunk_size":
            int(
                policy.config.chunk_size
            ),
        "postprocessed_action_dim":
            6,
        "action_names":
            list(
                ACTION_NAMES
            ),
        "actuator_ctrl_ranges":
            ctrl_ranges.tolist(),
        "same_noise_tensor_within_pair":
            True,
        "heldout_minus5_used_for_targets":
            False,
        "output_shapes": {
            key:
                list(
                    value.shape
                )
            for key, value
            in npz_payload.items()
            if isinstance(
                value,
                np.ndarray,
            )
        },
        "total_target_clip_counts": {
            offset_key(delta):
                int(
                    np.count_nonzero(
                        clip_masks[
                            delta
                        ]
                    )
                )
            for delta
            in perturb_offsets
        },
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

        f.write("\n")

    print()
    print(
        "TEACHER RESPONSE EXTRACTION PASSED"
    )

    print(
        "output:",
        output_dir,
    )

    for delta in perturb_offsets:
        print(
            f"delta={delta:+.1f}deg "
            f"mean_response_rms="
            f"{rms(mean_responses[delta]):.6f} "
            f"seed_dispersion_rms="
            f"{rms(responses[delta] - mean_responses[delta][:, None]):.6f} "
            f"clip_count="
            f"{np.count_nonzero(clip_masks[delta])}"
        )


if __name__ == "__main__":
    main()
