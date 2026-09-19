#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = (
    ROOT
    / "results/evaluation/smolvla_baseline_v1"
)

SEEDS = [
    20260920,
    20260921,
    20260922,
    20260923,
    20260924,
]

OUTPUT_DIR = (
    BASE_DIR
    / "action_response_5seeds_20variations"
)

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_seed_responses = []
    object_x_ref = None
    variation_ids_ref = None

    print("=== load results ===")

    for seed in SEEDS:
        path = (
            BASE_DIR
            / f"action_response_all_variations_seed{seed}"
            / "responses_all.npz"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing seed result: {path}"
            )

        data = np.load(path)

        responses = data["responses"]
        object_x = data["object_x_m"]
        variation_ids = data["variation_ids"]

        # expected:
        # variation x position x chunk x action
        if responses.shape != (20, 7, 50, 6):
            raise RuntimeError(
                f"Unexpected shape for seed {seed}: "
                f"{responses.shape}"
            )

        if object_x_ref is None:
            object_x_ref = object_x.copy()
            variation_ids_ref = variation_ids.copy()
        else:
            if not np.allclose(
                object_x,
                object_x_ref,
            ):
                raise RuntimeError(
                    f"Position mismatch at seed {seed}"
                )

            if not np.array_equal(
                variation_ids,
                variation_ids_ref,
            ):
                raise RuntimeError(
                    f"Variation mismatch at seed {seed}"
                )

        all_seed_responses.append(
            responses.astype(np.float64)
        )

        print(
            f"seed={seed} "
            f"shape={responses.shape}"
        )

    responses = np.stack(
        all_seed_responses,
        axis=0,
    )

    # shape:
    # seed x variation x position x chunk x action
    print("\ncombined shape:", responses.shape)

    delta_cm = (
        object_x_ref - 0.240
    ) * 100.0

    # --------------------------------------------------
    # Per seed x variation scalar RMS
    # --------------------------------------------------

    pair_rms = np.sqrt(
        np.mean(
            responses ** 2,
            axis=(3, 4),
        )
    )

    # shape:
    # seed x variation x position

    # --------------------------------------------------
    # Overall aggregate over 5 seeds x 20 variations
    # --------------------------------------------------

    rows = []

    print("\n=== overall: 5 seeds x 20 variations ===")

    for pos_idx, delta in enumerate(
        delta_cm
    ):
        values = pair_rms[
            :,
            :,
            pos_idx,
        ].reshape(-1)

        row = {
            "object_x_m":
                float(object_x_ref[pos_idx]),
            "delta_x_cm":
                float(delta),
            "n":
                int(values.size),
            "response_rms_mean":
                float(np.mean(values)),
            "response_rms_std":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
            "response_rms_median":
                float(np.median(values)),
            "response_rms_min":
                float(np.min(values)),
            "response_rms_max":
                float(np.max(values)),
        }

        rows.append(row)

        print(
            f"delta={delta:+.1f} cm "
            f"RMS={row['response_rms_mean']:.6f} "
            f"+/- {row['response_rms_std']:.6f} "
            f"median={row['response_rms_median']:.6f} "
            f"n={row['n']}"
        )

    overall_csv = (
        OUTPUT_DIR
        / "aggregate_100pairs.csv"
    )

    with overall_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Seed-level statistics
    # Each seed is first averaged over its 20 variations.
    # --------------------------------------------------

    seed_rows = []

    print("\n=== seed-level means ===")

    seed_mean_curve = np.mean(
        pair_rms,
        axis=1,
    )

    # seed x position

    for seed_idx, seed in enumerate(SEEDS):
        for pos_idx, delta in enumerate(
            delta_cm
        ):
            seed_rows.append(
                {
                    "seed": seed,
                    "object_x_m":
                        float(
                            object_x_ref[pos_idx]
                        ),
                    "delta_x_cm":
                        float(delta),
                    "response_rms_mean_over_variations":
                        float(
                            seed_mean_curve[
                                seed_idx,
                                pos_idx,
                            ]
                        ),
                }
            )

    for pos_idx, delta in enumerate(
        delta_cm
    ):
        values = seed_mean_curve[
            :,
            pos_idx,
        ]

        print(
            f"delta={delta:+.1f} cm "
            f"seed_mean="
            f"{np.mean(values):.6f} "
            f"+/- "
            f"{np.std(values, ddof=1):.6f}"
        )

    seed_csv = (
        OUTPUT_DIR
        / "seed_level_summary.csv"
    )

    with seed_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                seed_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    # --------------------------------------------------
    # Per-joint response
    # --------------------------------------------------

    # RMS across 50 chunk steps
    joint_rms = np.sqrt(
        np.mean(
            responses ** 2,
            axis=3,
        )
    )

    # seed x variation x position x action

    joint_rows = []

    print("\n=== per-joint aggregate ===")

    for pos_idx, delta in enumerate(
        delta_cm
    ):
        print(f"\ndelta={delta:+.1f} cm")

        for action_idx, name in enumerate(
            ACTION_NAMES
        ):
            values = joint_rms[
                :,
                :,
                pos_idx,
                action_idx,
            ].reshape(-1)

            mean = float(np.mean(values))
            std = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            joint_rows.append(
                {
                    "object_x_m":
                        float(
                            object_x_ref[
                                pos_idx
                            ]
                        ),
                    "delta_x_cm":
                        float(delta),
                    "action_name":
                        name,
                    "n":
                        int(values.size),
                    "rms_mean":
                        mean,
                    "rms_std":
                        std,
                    "rms_median":
                        float(
                            np.median(values)
                        ),
                }
            )

            print(
                f"  {name:14s} "
                f"{mean:.6f} +/- "
                f"{std:.6f}"
            )

    joint_csv = (
        OUTPUT_DIR
        / "per_joint_aggregate.csv"
    )

    with joint_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                joint_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            joint_rows
        )

    # --------------------------------------------------
    # Preserve full aggregate tensor
    # --------------------------------------------------

    np.savez_compressed(
        OUTPUT_DIR
        / "responses_5seeds_20variations.npz",
        seeds=np.asarray(
            SEEDS,
            dtype=np.int64,
        ),
        variation_ids=variation_ids_ref,
        object_x_m=object_x_ref,
        delta_x_m=(
            object_x_ref - 0.240
        ),
        responses=responses.astype(
            np.float32
        ),
        pair_rms=pair_rms.astype(
            np.float32
        ),
    )

    print("\nSaved:")
    print(overall_csv)
    print(seed_csv)
    print(joint_csv)
    print(
        OUTPUT_DIR
        / "responses_5seeds_20variations.npz"
    )

    print()
    print(
        "SMOLVLA 5-SEED AGGREGATION PASSED"
    )


if __name__ == "__main__":
    main()
