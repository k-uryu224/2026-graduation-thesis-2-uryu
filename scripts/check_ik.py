"""Validate calibrated SO-101 grasp IK over all training positions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from so101_vla_response.configuration import (
    ConfigError,
    load_experiment_bundle,
)
from so101_vla_response.ik import SO101MinkIK
from so101_vla_response.simulation import (
    MujocoEnvironment,
    SimulationError,
)


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}

LOCAL_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

WORLD_DIRECTIONS = {
    "world_positive_x": np.array([1.0, 0.0, 0.0]),
    "world_negative_x": np.array([-1.0, 0.0, 0.0]),
    "world_positive_y": np.array([0.0, 1.0, 0.0]),
    "world_negative_y": np.array([0.0, -1.0, 0.0]),
    "world_positive_z": np.array([0.0, 0.0, 1.0]),
    "world_negative_z": np.array([0.0, 0.0, -1.0]),
}


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping: {path}"
        )

    return data


def direction_from_tilt(
    tilt_deg: float,
    reference_name: str,
    toward_name: str,
) -> np.ndarray:
    reference = WORLD_DIRECTIONS[
        reference_name
    ]

    toward = WORLD_DIRECTIONS[
        toward_name
    ]

    if not math.isclose(
        float(np.dot(reference, toward)),
        0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Tilt reference and toward directions "
            "must be orthogonal."
        )

    angle = math.radians(
        float(tilt_deg)
    )

    direction = (
        math.cos(angle) * reference
        + math.sin(angle) * toward
    )

    return direction / np.linalg.norm(
        direction
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/response_comparison.yaml",
    )
    parser.add_argument(
        "--scene-config",
        default="configs/scene/top_grasp.yaml",
    )
    parser.add_argument(
        "--task-config",
        default="configs/task/pick_object.yaml",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
    )

    args = parser.parse_args()

    try:
        bundle = load_experiment_bundle(
            Path(args.experiment_config)
        )

        env = MujocoEnvironment.from_robot_config(
            bundle.section("robot").root,
            repo_root=Path(args.repo_root),
        )

        env.reset()

    except (
        ConfigError,
        SimulationError,
    ) as exc:
        print(f"IK check failed: {exc}")
        return 1

    scene = load_yaml(
        Path(args.scene_config)
    )["scene"]

    task = load_yaml(
        Path(args.task_config)
    )["task"]

    object_site_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    if object_site_id < 0:
        print(
            "IK check failed: "
            "target_box_center not found."
        )
        return 1

    baseline_object_position = (
        env.data.site_xpos[
            object_site_id
        ].copy()
    )

    perturbation = scene[
        "perturbation"
    ]

    perturb_axis = perturbation[
        "axis"
    ]

    perturb_index = AXIS_TO_INDEX[
        perturb_axis
    ]

    offsets = perturbation[
        "training_offsets_m"
    ]

    tcp = task["tcp"]

    local_axis_name = tcp[
        "approach_axis"
    ]["local_axis"]

    local_axis = LOCAL_AXES[
        local_axis_name
    ]

    tilt_reference = tcp[
        "approach_axis"
    ]["tilt_reference"]

    tilt_toward = tcp[
        "approach_axis"
    ]["tilt_toward"]

    target_definitions = tcp[
        "targets"
    ]

    phase_names = (
        "pre_grasp",
        "grasp",
        "lift",
    )

    solver = SO101MinkIK(
        model=env.model,
        tcp_site=tcp["site"],
        local_approach_axis=local_axis,
    )

    print("SO-101 calibrated Mink IK check")
    print()
    print(
        "Arm DOFs:",
        solver.arm_dof_indices,
    )
    print(
        "Frozen DOFs:",
        solver.frozen_dof_indices,
    )
    print(
        "Baseline object:",
        np.round(
            baseline_object_position,
            6,
        ),
    )
    print(
        "Perturbation axis:",
        f"world-{perturb_axis}",
    )

    all_reached = True

    for object_offset in offsets:
        object_position = (
            baseline_object_position.copy()
        )

        object_position[
            perturb_index
        ] += float(object_offset)

        qpos = env.data.qpos.copy()

        print()
        print("=" * 72)
        print(
            "Object offset:",
            f"{float(object_offset) * 1000:+.1f} mm",
        )
        print(
            "Object position:",
            np.round(
                object_position,
                6,
            ),
        )

        for phase_name in phase_names:
            phase = target_definitions[
                phase_name
            ]

            relative_offset = np.asarray(
                phase["offset_m"],
                dtype=float,
            )

            target_position = (
                object_position
                + relative_offset
            )

            tilt_deg = float(
                phase[
                    "approach_tilt_deg"
                ]
            )

            target_direction = (
                direction_from_tilt(
                    tilt_deg,
                    tilt_reference,
                    tilt_toward,
                )
            )

            result = solver.solve(
                qpos,
                target_position,
                target_direction,
            )

            print()
            print(
                f"[{phase_name}]"
            )
            print(
                "  target:",
                np.round(
                    target_position,
                    6,
                ),
            )
            print(
                f"  tilt: "
                f"{tilt_deg:.1f} deg"
            )
            print(
                f"  reached: "
                f"{result.reached}"
            )
            print(
                f"  iterations: "
                f"{result.iterations}"
            )
            print(
                f"  position error: "
                f"{result.position_error_mm:.3f} mm"
            )
            print(
                f"  axis error: "
                f"{result.axis_error_deg:.3f} deg"
            )
            print(
                f"  min joint margin: "
                f"{result.min_joint_margin_deg:.3f} deg"
            )

            if not result.reached:
                all_reached = False
                break

            qpos = result.qpos.copy()

    print()
    print("=" * 72)

    if all_reached:
        print(
            "IK CHECK PASSED: "
            "all calibrated training targets are reachable."
        )
        return 0

    print(
        "IK CHECK FAILED: "
        "at least one calibrated target did not converge."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
