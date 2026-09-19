#!/usr/bin/env python3

import argparse
import csv
import importlib.util
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
        "action_response_base",
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
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA action responses across fixed variations."
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        required=True,
        help="Flow-matching noise seed shared across all positions and variations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    noise_seed = args.noise_seed

    base = load_base_module()

    output_dir = (
        ROOT
        / "results/evaluation/smolvla_baseline_v1"
        / f"action_response_all_variations_seed{noise_seed}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda")

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

    print("=== evaluation ===")
    print("variations:", len(variations))
    print(
        "positions:",
        base.OBJECT_X_M.tolist(),
    )
    print("noise seed:", noise_seed)

    # ----------------------------------------
    # Policy
    # ----------------------------------------

    print("\n=== load policy ===")

    policy = base.SmolVLAPolicy.from_pretrained(
        str(base.CHECKPOINT)
    )
    policy = policy.to(device)
    policy.eval()

    (
        preprocessor,
        postprocessor,
    ) = base.make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(base.CHECKPOINT),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
            "rename_observations_processor": {
                "rename_map": base.RENAME_MAP,
            },
        },
    )

    # ----------------------------------------
    # Common random noise
    # ----------------------------------------

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        noise_seed
    )

    fixed_noise = torch.randn(
        noise_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
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

    all_actions = []
    all_responses = []
    all_states = []

    rows = []

    baseline_idx = int(
        np.where(
            np.isclose(
                base.OBJECT_X_M,
                base.BASELINE_X_M,
                atol=1e-9,
            )
        )[0][0]
    )

    try:
        for variation_index, variation in enumerate(
            variations
        ):
            variation_id = variation["id"]

            print()
            print(
                "=" * 70
            )
            print(
                f"{variation_index + 1:02d}/"
                f"{len(variations):02d} "
                f"{variation_id}"
            )

            states = []
            action_chunks = []

            for object_x in base.OBJECT_X_M:
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

                raw_state = (
                    observation[
                        "observation.state"
                    ]
                    .copy()
                    .astype(np.float32)
                )

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
                            noise=fixed_noise.clone(),
                        )
                    )

                    chunk = postprocessor(
                        raw_chunk.clone()
                    )

                chunk_np = (
                    chunk[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                states.append(raw_state)
                action_chunks.append(
                    chunk_np
                )

                print(
                    f"  x={object_x:.3f}"
                )

            states = np.stack(states)
            action_chunks = np.stack(
                action_chunks
            )

            # Every x must have identical robot state
            state_diff = np.max(
                np.abs(
                    states
                    - states[baseline_idx]
                ),
                axis=1,
            )

            if np.max(state_diff) > 1e-6:
                raise RuntimeError(
                    f"{variation_id}: "
                    f"state differs across x: "
                    f"{state_diff}"
                )

            responses = (
                action_chunks
                - action_chunks[baseline_idx]
            )

            all_states.append(states)
            all_actions.append(
                action_chunks
            )
            all_responses.append(
                responses
            )

            for pos_idx, object_x in enumerate(
                base.OBJECT_X_M
            ):
                response = responses[pos_idx]

                per_joint_rms = np.sqrt(
                    np.mean(
                        response ** 2,
                        axis=0,
                    )
                )

                abs_response = np.abs(
                    response
                )

                flat_index = int(
                    np.argmax(
                        abs_response
                    )
                )

                (
                    max_step,
                    max_action_idx,
                ) = np.unravel_index(
                    flat_index,
                    response.shape,
                )

                row = {
                    "variation_id":
                        variation_id,
                    "object_x_m":
                        float(object_x),
                    "delta_x_cm":
                        float(
                            (
                                object_x
                                - base.BASELINE_X_M
                            )
                            * 100.0
                        ),
                    "response_rms":
                        float(
                            np.sqrt(
                                np.mean(
                                    response ** 2
                                )
                            )
                        ),
                    "response_mean_abs":
                        float(
                            np.mean(
                                np.abs(
                                    response
                                )
                            )
                        ),
                    "first_action_l2":
                        float(
                            np.linalg.norm(
                                response[0]
                            )
                        ),
                    "max_abs_response":
                        float(
                            abs_response[
                                max_step,
                                max_action_idx,
                            ]
                        ),
                    "max_step":
                        int(max_step),
                    "max_action":
                        base.ACTION_NAMES[
                            max_action_idx
                        ],
                }

                for action_idx, name in enumerate(
                    base.ACTION_NAMES
                ):
                    row[
                        f"rms_{name}"
                    ] = float(
                        per_joint_rms[
                            action_idx
                        ]
                    )

                rows.append(row)

    finally:
        renderer.close()

    all_states = np.stack(
        all_states
    )
    all_actions = np.stack(
        all_actions
    )
    all_responses = np.stack(
        all_responses
    )

    # shape:
    # variations x positions x chunk x action
    np.savez_compressed(
        output_dir / "responses_all.npz",
        object_x_m=base.OBJECT_X_M,
        delta_x_m=(
            base.OBJECT_X_M
            - base.BASELINE_X_M
        ),
        states=all_states,
        action_chunks=all_actions,
        responses=all_responses,
        variation_ids=np.asarray(
            [
                variation["id"]
                for variation in variations
            ]
        ),
        noise_seed=np.asarray(
            noise_seed,
            dtype=np.int64,
        ),
        fixed_noise=(
            fixed_noise
            .detach()
            .cpu()
            .numpy()
        ),
    )

    detail_csv = (
        output_dir
        / "variation_summary.csv"
    )

    with detail_csv.open(
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

    # ----------------------------------------
    # Aggregate across variations
    # ----------------------------------------

    aggregate_rows = []

    print()
    print(
        "=== aggregate over variations ==="
    )

    for pos_idx, object_x in enumerate(
        base.OBJECT_X_M
    ):
        r = all_responses[
            :,
            pos_idx,
            :,
            :,
        ]

        rms_each_variation = np.sqrt(
            np.mean(
                r ** 2,
                axis=(1, 2),
            )
        )

        first_l2_each = np.linalg.norm(
            r[:, 0, :],
            axis=1,
        )

        joint_rms_each = np.sqrt(
            np.mean(
                r ** 2,
                axis=1,
            )
        )

        row = {
            "object_x_m":
                float(object_x),
            "delta_x_cm":
                float(
                    (
                        object_x
                        - base.BASELINE_X_M
                    )
                    * 100.0
                ),
            "response_rms_mean":
                float(
                    np.mean(
                        rms_each_variation
                    )
                ),
            "response_rms_std":
                float(
                    np.std(
                        rms_each_variation,
                        ddof=1,
                    )
                )
                if len(variations) > 1
                else 0.0,
            "response_rms_median":
                float(
                    np.median(
                        rms_each_variation
                    )
                ),
            "first_action_l2_mean":
                float(
                    np.mean(
                        first_l2_each
                    )
                ),
        }

        for action_idx, name in enumerate(
            base.ACTION_NAMES
        ):
            row[
                f"rms_{name}_mean"
            ] = float(
                np.mean(
                    joint_rms_each[
                        :,
                        action_idx,
                    ]
                )
            )

            row[
                f"rms_{name}_std"
            ] = float(
                np.std(
                    joint_rms_each[
                        :,
                        action_idx,
                    ],
                    ddof=1,
                )
            )

        aggregate_rows.append(row)

        print(
            f"delta="
            f"{row['delta_x_cm']:+.1f} cm "
            f"RMS="
            f"{row['response_rms_mean']:.6f} "
            f"+/- "
            f"{row['response_rms_std']:.6f} "
            f"median="
            f"{row['response_rms_median']:.6f}"
        )

    aggregate_csv = (
        output_dir
        / "aggregate_summary.csv"
    )

    with aggregate_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                aggregate_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            aggregate_rows
        )

    print()
    print("Saved:")
    print(
        output_dir
        / "responses_all.npz"
    )
    print(detail_csv)
    print(aggregate_csv)

    print()
    print(
        "SMOLVLA ALL-VARIATION "
        "ACTION RESPONSE PASSED"
    )


if __name__ == "__main__":
    main()
