"""Validate scripted SO-101 grasp trajectories at all training positions."""

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
from so101_vla_response.ik.mink_solver import ARM_JOINT_NAMES
from so101_vla_response.simulation import (
    MujocoEnvironment,
    SimulationError,
)
from so101_vla_response.trajectory.scripted_grasp import (
    DEFAULT_RATE_HZ,
    build_scripted_grasp_trajectory,
    phase_frame_counts,
    phase_slices,
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


def pinch_direction_from_approach(
    approach_direction: np.ndarray,
) -> np.ndarray:
    """Project world +X perpendicular to the approach axis."""

    approach = np.asarray(
        approach_direction,
        dtype=float,
    )

    approach /= np.linalg.norm(
        approach
    )

    reference = np.array(
        [1.0, 0.0, 0.0],
        dtype=float,
    )

    pinch = (
        reference
        - np.dot(
            reference,
            approach,
        )
        * approach
    )

    norm = float(
        np.linalg.norm(
            pinch
        )
    )

    if norm <= 1e-9:
        raise RuntimeError(
            "Could not construct pinch direction."
        )

    return pinch / norm


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


def extract_arm_qpos(
    model: mujoco.MjModel,
    full_qpos: np.ndarray,
) -> np.ndarray:
    values: list[float] = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise ValueError(
                f"Arm joint not found: {joint_name}"
            )

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        values.append(
            float(full_qpos[qpos_address])
        )

    return np.asarray(
        values,
        dtype=float,
    )


def check_arm_joint_limits(
    model: mujoco.MjModel,
    arm_trajectory: np.ndarray,
) -> tuple[bool, float]:
    minimum_margin = float("inf")

    for arm_index, joint_name in enumerate(
        ARM_JOINT_NAMES
    ):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        lower = float(
            model.jnt_range[joint_id][0]
        )
        upper = float(
            model.jnt_range[joint_id][1]
        )

        values = arm_trajectory[
            :,
            arm_index,
        ]

        margins = np.minimum(
            values - lower,
            upper - values,
        )

        joint_minimum = float(
            np.min(margins)
        )

        minimum_margin = min(
            minimum_margin,
            joint_minimum,
        )

        if joint_minimum < -1e-9:
            return False, minimum_margin

    return True, minimum_margin


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
        print(
            f"Trajectory check failed: {exc}"
        )
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
            "Trajectory check failed: "
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

    local_axis = LOCAL_AXES[
        tcp["approach_axis"][
            "local_axis"
        ]
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

    pinch_axis_config = tcp[
        "pinch_axis"
    ]

    if (
        pinch_axis_config[
            "local_axis"
        ]
        != "z"
    ):
        raise ValueError(
            "Only local +Z pinch axis is "
            "currently supported."
        )

    if (
        pinch_axis_config[
            "reference_axis"
        ]
        != "world_positive_x"
    ):
        raise ValueError(
            "Only world_positive_x pinch "
            "reference is currently supported."
        )

    solver = SO101MinkIK(
        model=env.model,
        tcp_site=tcp["site"],
        local_approach_axis=local_axis,
            local_pinch_axis=np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        ),
        pinch_cost=float(
            pinch_axis_config[
                "cost"
            ]
        ),
)

    home_full_qpos = env.data.qpos.copy()

    home_arm_qpos = extract_arm_qpos(
        env.model,
        home_full_qpos,
    )

    expected_counts = phase_frame_counts(
        DEFAULT_RATE_HZ
    )

    slices = phase_slices(
        DEFAULT_RATE_HZ
    )

    expected_total_frames = sum(
        expected_counts.values()
    )

    overall_ok = True

    print("SO-101 scripted trajectory check")
    print()
    print(
        f"rate: {DEFAULT_RATE_HZ:.1f} Hz"
    )
    print(
        f"expected frames: "
        f"{expected_total_frames}"
    )
    print(
        f"expected duration: "
        f"{expected_total_frames / DEFAULT_RATE_HZ:.1f} s"
    )

    print()
    print("phase layout:")

    for phase_name, phase_slice in slices.items():
        print(
            f"  {phase_name:<22}"
            f"{phase_slice.start:3d}"
            f"..{phase_slice.stop - 1:3d} "
            f"({phase_slice.stop - phase_slice.start:3d} frames)"
        )

    for object_offset in offsets:
        object_position = (
            baseline_object_position.copy()
        )

        object_position[
            perturb_index
        ] += float(object_offset)

        qpos = home_full_qpos.copy()

        waypoint_arm_qpos: dict[
            str,
            np.ndarray,
        ] = {}

        ik_ok = True

        # ----------------------------------------------------
        # 1. Pre-grasp
        # ----------------------------------------------------

        pre_phase = target_definitions[
            "pre_grasp"
        ]

        pre_target = (
            object_position
            + np.asarray(
                pre_phase[
                    "offset_m"
                ],
                dtype=float,
            )
        )

        pre_direction = (
            direction_from_tilt(
                float(
                    pre_phase[
                        "approach_tilt_deg"
                    ]
                ),
                tilt_reference,
                tilt_toward,
            )
        )

        pre_result = solver.solve(
            qpos,
            pre_target,
            pre_direction,
        )

        if not pre_result.reached:
            ik_ok = False
            overall_ok = False

        if ik_ok:
            waypoint_arm_qpos[
                "pre_grasp"
            ] = extract_arm_qpos(
                env.model,
                pre_result.qpos,
            )

            qpos = (
                pre_result.qpos.copy()
            )

        # ----------------------------------------------------
        # Lift definitions shared by approach and lift.
        # ----------------------------------------------------

        lift_definitions = task[
            "trajectory"
        ][
            "lift_waypoints"
        ]

        lift_by_name = {
            phase[
                "name"
            ]: phase
            for phase
            in lift_definitions
        }

        approach_order = task[
            "trajectory"
        ][
            "approach"
        ][
            "waypoint_order"
        ]

        if (
            not approach_order
            or approach_order[-1]
            != "grasp"
        ):
            raise RuntimeError(
                "Approach waypoint order must "
                "terminate at grasp."
            )

        approach_waypoint_arm_qpos = []

        # ----------------------------------------------------
        # 2. Reverse-Lift approach:
        #    Lift-6 -> ... -> Lift-1
        # ----------------------------------------------------

        if ik_ok:
            for waypoint_name in (
                approach_order[:-1]
            ):
                if waypoint_name not in lift_by_name:
                    raise RuntimeError(
                        f"Unknown approach waypoint: "
                        f"{waypoint_name}"
                    )

                phase = lift_by_name[
                    waypoint_name
                ]

                target_position = (
                    object_position
                    + np.asarray(
                        phase[
                            "offset_m"
                        ],
                        dtype=float,
                    )
                )

                target_direction = (
                    direction_from_tilt(
                        float(
                            phase[
                                "approach_tilt_deg"
                            ]
                        ),
                        tilt_reference,
                        tilt_toward,
                    )
                )

                target_pinch_direction = (
                    pinch_direction_from_approach(
                        target_direction
                    )
                )

                result = solver.solve(
                    qpos,
                    target_position,
                    target_direction,
                    target_pinch_direction=(
                        target_pinch_direction
                    ),
                )

                if not result.reached:
                    ik_ok = False
                    overall_ok = False
                    break

                arm_qpos = extract_arm_qpos(
                    env.model,
                    result.qpos,
                )

                waypoint_arm_qpos[
                    waypoint_name
                ] = arm_qpos

                approach_waypoint_arm_qpos.append(
                    arm_qpos
                )

                qpos = (
                    result.qpos.copy()
                )

        # ----------------------------------------------------
        # 3. Grasp
        # ----------------------------------------------------

        if ik_ok:
            grasp_phase = (
                target_definitions[
                    "grasp"
                ]
            )

            grasp_target = (
                object_position
                + np.asarray(
                    grasp_phase[
                        "offset_m"
                    ],
                    dtype=float,
                )
            )

            grasp_direction = (
                direction_from_tilt(
                    float(
                        grasp_phase[
                            "approach_tilt_deg"
                        ]
                    ),
                    tilt_reference,
                    tilt_toward,
                )
            )

            grasp_pinch_direction = (
                pinch_direction_from_approach(
                    grasp_direction
                )
            )

            grasp_result = solver.solve(
                qpos,
                grasp_target,
                grasp_direction,
                target_pinch_direction=(
                    grasp_pinch_direction
                ),
            )

            if not grasp_result.reached:
                ik_ok = False
                overall_ok = False

            else:
                waypoint_arm_qpos[
                    "grasp"
                ] = extract_arm_qpos(
                    env.model,
                    grasp_result.qpos,
                )

                qpos = (
                    grasp_result.qpos.copy()
                )

        # ----------------------------------------------------
        # 4. Forward Lift-1 -> Lift-8
        # ----------------------------------------------------

        lift_waypoint_arm_qpos = []

        if ik_ok:
            for phase in lift_definitions:
                waypoint_name = (
                    phase[
                        "name"
                    ]
                )

                target_position = (
                    object_position
                    + np.asarray(
                        phase[
                            "offset_m"
                        ],
                        dtype=float,
                    )
                )

                target_direction = (
                    direction_from_tilt(
                        float(
                            phase[
                                "approach_tilt_deg"
                            ]
                        ),
                        tilt_reference,
                        tilt_toward,
                    )
                )

                target_pinch_direction = (
                    pinch_direction_from_approach(
                        target_direction
                    )
                )

                result = solver.solve(
                    qpos,
                    target_position,
                    target_direction,
                    target_pinch_direction=(
                        target_pinch_direction
                    ),
                )

                if not result.reached:
                    ik_ok = False
                    overall_ok = False
                    break

                arm_qpos = extract_arm_qpos(
                    env.model,
                    result.qpos,
                )

                waypoint_arm_qpos[
                    waypoint_name
                ] = arm_qpos

                lift_waypoint_arm_qpos.append(
                    arm_qpos
                )

                qpos = (
                    result.qpos.copy()
                )

        # Compatibility alias used by existing checks.
        if ik_ok:
            waypoint_arm_qpos[
                "lift"
            ] = (
                lift_waypoint_arm_qpos[
                    -1
                ]
            )

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

        if not ik_ok:
            print(
                "  FAIL: waypoint IK did not converge."
            )
            continue

        trajectory = (
            build_scripted_grasp_trajectory(
                home_arm_qpos=home_arm_qpos,
                pre_grasp_arm_qpos=waypoint_arm_qpos[
                    "pre_grasp"
                ],
                grasp_arm_qpos=waypoint_arm_qpos[
                    "grasp"
                ],
                lift_arm_qpos=waypoint_arm_qpos[
                    "lift"
                ],
                approach_waypoint_arm_qpos=(
                    approach_waypoint_arm_qpos
                ),
                lift_waypoint_arm_qpos=(
                    lift_waypoint_arm_qpos
                ),
                rate_hz=DEFAULT_RATE_HZ,
            )
        )

        checks: list[
            tuple[str, bool]
        ] = []

        checks.append(
            (
                "frame_count",
                trajectory.frame_count
                == expected_total_frames,
            )
        )

        checks.append(
            (
                "duration",
                math.isclose(
                    trajectory.duration_s,
                    7.0,
                    abs_tol=1e-12,
                ),
            )
        )

        checks.append(
            (
                "phase_counts",
                trajectory.phase_counts()
                == expected_counts,
            )
        )

        checks.append(
            (
                "initial_home",
                np.allclose(
                    trajectory.arm_qpos[
                        slices[
                            "initial_hold"
                        ].start
                    ],
                    home_arm_qpos,
                    atol=1e-10,
                ),
            )
        )

        checks.append(
            (
                "pre_grasp_endpoint",
                np.allclose(
                    trajectory.arm_qpos[
                        slices[
                            "home_to_pre_grasp"
                        ].stop
                        - 1
                    ],
                    waypoint_arm_qpos[
                        "pre_grasp"
                    ],
                    atol=1e-10,
                ),
            )
        )

        checks.append(
            (
                "grasp_endpoint",
                np.allclose(
                    trajectory.arm_qpos[
                        slices[
                            "pre_grasp_to_grasp"
                        ].stop
                        - 1
                    ],
                    waypoint_arm_qpos[
                        "grasp"
                    ],
                    atol=1e-10,
                ),
            )
        )

        checks.append(
            (
                "lift_endpoint",
                np.allclose(
                    trajectory.arm_qpos[
                        slices[
                            "grasp_to_lift"
                        ].stop
                        - 1
                    ],
                    waypoint_arm_qpos[
                        "lift"
                    ],
                    atol=1e-10,
                ),
            )
        )

        close_slice = slices[
            "gripper_close"
        ]

        checks.append(
            (
                "gripper_open_before_close",
                math.isclose(
                    float(
                        trajectory.gripper_close_fraction[
                            close_slice.start - 1
                        ]
                    ),
                    0.0,
                    abs_tol=1e-12,
                ),
            )
        )

        checks.append(
            (
                "gripper_closed_after_close",
                math.isclose(
                    float(
                        trajectory.gripper_close_fraction[
                            close_slice.stop - 1
                        ]
                    ),
                    1.0,
                    abs_tol=1e-12,
                ),
            )
        )

        limits_ok, min_margin = (
            check_arm_joint_limits(
                env.model,
                trajectory.arm_qpos,
            )
        )

        checks.append(
            (
                "joint_limits",
                limits_ok,
            )
        )

        max_step_rad = float(
            np.max(
                np.abs(
                    np.diff(
                        trajectory.arm_qpos,
                        axis=0,
                    )
                )
            )
        )

        max_step_deg = math.degrees(
            max_step_rad
        )

        local_ok = all(
            passed
            for _, passed in checks
        )

        if not local_ok:
            overall_ok = False

        for name, passed in checks:
            print(
                f"  {name:<28}"
                f"{'PASS' if passed else 'FAIL'}"
            )

        print(
            "  minimum joint margin      "
            f"{math.degrees(min_margin):.3f} deg"
        )
        print(
            "  maximum arm step/frame    "
            f"{max_step_deg:.3f} deg"
        )

    print()
    print("=" * 72)

    if overall_ok:
        print(
            "TRAJECTORY CHECK PASSED: "
            "all training positions produce valid "
            "210-frame trajectories."
        )
        return 0

    print(
        "TRAJECTORY CHECK FAILED: "
        "at least one trajectory validation failed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
