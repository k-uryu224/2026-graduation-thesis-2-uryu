#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    ROOT
    / "results/evaluation/smolvla_baseline_v1"
    / "action_response_v00_seed20260920"
)

NPZ_PATH = RESULT_DIR / "action_response.npz"

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    data = np.load(NPZ_PATH)

    object_x = data["object_x_m"]
    delta_cm = data["delta_x_m"] * 100.0
    actions = data["action_chunks"]
    responses = data["responses"]

    # ----------------------------------------
    # 1. Whole-chunk RMS response
    # ----------------------------------------

    rms = np.sqrt(
        np.mean(
            responses ** 2,
            axis=(1, 2),
        )
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        delta_cm,
        rms,
        marker="o",
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel("Object displacement from baseline [cm]")
    ax.set_ylabel("Action response RMS")
    ax.set_title("SmolVLA Action Response")
    ax.grid(True)

    fig.tight_layout()

    out = RESULT_DIR / "response_rms_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print("saved:", out)

    # ----------------------------------------
    # 2. Response for each joint over 50 steps
    # ----------------------------------------

    steps = np.arange(
        responses.shape[1]
    )

    for action_index, action_name in enumerate(
        ACTION_NAMES
    ):
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for pos_index, delta in enumerate(
            delta_cm
        ):
            if np.isclose(delta, 0.0):
                continue

            ax.plot(
                steps,
                responses[
                    pos_index,
                    :,
                    action_index,
                ],
                label=f"{delta:+.1f} cm",
            )

        ax.axhline(
            0.0,
            linewidth=1,
        )

        ax.set_xlabel("Action chunk step")
        ax.set_ylabel("Response")
        ax.set_title(
            f"Action Response: {action_name}"
        )
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        out = (
            RESULT_DIR
            / f"response_{action_name}.png"
        )

        fig.savefig(
            out,
            dpi=180,
        )
        plt.close(fig)

        print("saved:", out)

    # ----------------------------------------
    # 3. Per-joint RMS vs displacement
    # ----------------------------------------

    per_joint_rms = np.sqrt(
        np.mean(
            responses ** 2,
            axis=1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for action_index, action_name in enumerate(
        ACTION_NAMES
    ):
        ax.plot(
            delta_cm,
            per_joint_rms[:, action_index],
            marker="o",
            label=action_name,
        )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Object displacement from baseline [cm]"
    )
    ax.set_ylabel(
        "Per-joint response RMS"
    )
    ax.set_title(
        "SmolVLA Per-Joint Action Response"
    )
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    out = (
        RESULT_DIR
        / "response_per_joint_rms.png"
    )

    fig.savefig(
        out,
        dpi=180,
    )

    plt.close(fig)

    print("saved:", out)

    # ----------------------------------------
    # Console diagnostics
    # ----------------------------------------

    print()
    print("=== per-joint RMS ===")

    header = (
        "delta_cm "
        + " ".join(
            f"{name:>14s}"
            for name in ACTION_NAMES
        )
    )

    print(header)

    for pos_index, delta in enumerate(
        delta_cm
    ):
        values = " ".join(
            f"{value:14.6f}"
            for value in per_joint_rms[
                pos_index
            ]
        )

        print(
            f"{delta:+7.1f} {values}"
        )

    print()
    print("=== maximum absolute response ===")

    abs_response = np.abs(responses)

    for pos_index, delta in enumerate(
        delta_cm
    ):
        flat_index = np.argmax(
            abs_response[pos_index]
        )

        step_index, action_index = (
            np.unravel_index(
                flat_index,
                abs_response[
                    pos_index
                ].shape,
            )
        )

        value = responses[
            pos_index,
            step_index,
            action_index,
        ]

        print(
            f"{delta:+.1f} cm: "
            f"{ACTION_NAMES[action_index]} "
            f"step={step_index} "
            f"response={value:+.6f}"
        )


if __name__ == "__main__":
    main()
