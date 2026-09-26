#!/usr/bin/env python3

import csv
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    ROOT
    / "results/lprd_shoulder_lift_poc_v1/"
    "heldout_minus5"
)

PROTOCOL = (
    ROOT
    / "configs/evaluation/"
    "lprd_shoulder_lift_poc_v1_heldout_eval_protocol.json"
)

RESULT_LOCK = (
    ROOT
    / "configs/evaluation/locks/"
    "lprd_shoulder_lift_poc_v1_heldout_results.sha256"
)

REPORT_DIR = (
    ROOT
    / "reports/lprd_shoulder_lift_poc_v1"
)

POLICIES = [
    "frozen_base",
    "no_response",
    "pointwise_kd",
    "response_target",
]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1.0 + z * z / n

    center = (
        p + z * z / (2.0 * n)
    ) / d

    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / d
    )

    return (
        center - half,
        center + half,
    )


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only

    if n == 0:
        return 1.0

    m = min(a_only, b_only)

    lower = sum(
        math.comb(n, i)
        for i in range(m + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * lower,
    )


def read_policy(policy):
    path = (
        RESULT_ROOT
        / policy
        / "trial_summary.csv"
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    parsed = {}

    for row in rows:
        key = (
            row["variation_id"],
            int(row["noise_seed"]),
        )

        if key in parsed:
            raise RuntimeError(
                f"duplicate key: {policy} {key}"
            )

        parsed[key] = {
            "success":
                int(row["success"]),
            "max_lift_m":
                float(row["max_lift_m"]),
            "total_clipped_values":
                int(
                    row[
                        "total_clipped_values"
                    ]
                ),
        }

    if len(parsed) != 45:
        raise RuntimeError(
            f"{policy}: expected 45, "
            f"got {len(parsed)}"
        )

    return parsed


def main():
    subprocess.run(
        [
            "sha256sum",
            "-c",
            str(RESULT_LOCK),
        ],
        cwd=ROOT,
        check=True,
    )

    protocol = json.loads(
        PROTOCOL.read_text()
    )

    expected_keys = {
        (variation, seed)
        for variation
        in protocol["evaluation"][
            "variation_ids"
        ]
        for seed
        in protocol["evaluation"][
            "noise_seeds"
        ]
    }

    data = {
        policy: read_policy(policy)
        for policy in POLICIES
    }

    for policy in POLICIES:
        if set(data[policy]) != expected_keys:
            raise RuntimeError(
                f"matched-key mismatch: "
                f"{policy}"
            )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    for policy in POLICIES:
        rows = list(
            data[policy].values()
        )

        successes = sum(
            row["success"]
            for row in rows
        )

        n = len(rows)
        low, high = wilson(
            successes,
            n,
        )

        summary.append(
            {
                "policy": policy,
                "successes": successes,
                "n": n,
                "success_rate":
                    successes / n,
                "wilson95_low": low,
                "wilson95_high": high,
                "max_lift_mm_mean":
                    1000.0
                    * sum(
                        row["max_lift_m"]
                        for row in rows
                    )
                    / n,
                "clip_count_mean":
                    sum(
                        row[
                            "total_clipped_values"
                        ]
                        for row in rows
                    )
                    / n,
            }
        )

    comparisons = [
        (
            "response_target",
            "frozen_base",
            "primary",
        ),
        (
            "response_target",
            "no_response",
            "secondary",
        ),
        (
            "response_target",
            "pointwise_kd",
            "secondary",
        ),
    ]

    paired = []

    for a, b, role in comparisons:
        both_success = 0
        a_only = 0
        b_only = 0
        both_fail = 0

        for key in sorted(
            expected_keys
        ):
            sa = data[a][key][
                "success"
            ]

            sb = data[b][key][
                "success"
            ]

            if sa and sb:
                both_success += 1
            elif sa and not sb:
                a_only += 1
            elif not sa and sb:
                b_only += 1
            else:
                both_fail += 1

        paired.append(
            {
                "role": role,
                "policy_a": a,
                "policy_b": b,
                "both_success":
                    both_success,
                "a_only_success":
                    a_only,
                "b_only_success":
                    b_only,
                "both_fail":
                    both_fail,
                "discordant":
                    a_only + b_only,
                "mcnemar_exact_two_sided_p":
                    exact_mcnemar(
                        a_only,
                        b_only,
                    ),
            }
        )

    summary_path = (
        REPORT_DIR
        / "heldout_success_statistics.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary)

    paired_path = (
        REPORT_DIR
        / "heldout_mcnemar.csv"
    )

    with paired_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                paired[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(paired)

    report = {
        "experiment_id":
            protocol[
                "experiment_id"
            ],
        "matched_unit":
            "variation_id x noise_seed",
        "n_per_policy": 45,
        "success_statistics":
            summary,
        "paired_comparisons":
            paired,
        "interpretation_guardrails": [
            "The response-target PoC did not improve the held-out -5deg success endpoint.",
            "All three finetuned policies achieved zero successes, including the no-response control.",
            "Therefore this result does not isolate teacher-response supervision as the cause of failure.",
            "The current finetuning protocol should be treated as a candidate source of behavioral collapse.",
            "The held-out -5deg condition has now been observed and must not be reused as an untouched tuning target."
        ],
    }

    (
        REPORT_DIR
        / "heldout_statistics.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== success statistics ==="
    )

    for row in summary:
        print(
            f"{row['policy']:16s} "
            f"{row['successes']:2d}/"
            f"{row['n']:2d} "
            f"rate="
            f"{100*row['success_rate']:.1f}% "
            f"CI=["
            f"{100*row['wilson95_low']:.1f},"
            f"{100*row['wilson95_high']:.1f}] "
            f"max_lift_mean="
            f"{row['max_lift_mm_mean']:.1f}mm"
        )

    print()
    print(
        "=== paired exact McNemar ==="
    )

    for row in paired:
        print(
            f"{row['policy_a']} vs "
            f"{row['policy_b']}: "
            f"a_only={row['a_only_success']} "
            f"b_only={row['b_only_success']} "
            f"p="
            f"{row['mcnemar_exact_two_sided_p']:.6f}"
        )


if __name__ == "__main__":
    main()
