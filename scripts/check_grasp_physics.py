"""Validate scripted SO-101 grasping with MuJoCo physics."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
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

TARGET_BOX_BODY_NAME = "target_box"
TARGET_BOX_FREEJOINT_NAME = "target_box_freejoint"

GRIPPER_LINK_BODY_NAME = "gripper_link"
JAW_LINK_BODY_NAME = "jaw_link"

GRIPPER_JOINT_NAME = "gripper_joint"


@dataclass(frozen=True)
class MappingResult:
    mapping_name: str
    offset_m: float
    success: bool
    initial_object_z: float
    final_object_z: float
    max_object_z: float
    final_lift_m: float
    max_lift_m: float
    hold_frames: int
    required_hold_frames: int
    moving_jaw_contact_seen: bool
    fixed_finger_contact_seen: bool
    simultaneous_two_side_contact_seen: bool
    pre_close_contact_seen: bool
    max_gripper_object_contacts: int
    simulation_time_s: float


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


def object_geom_ids(
    model: mujoco.MjModel,
    body_ids: set[int],
) -> set[int]:
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def descendant_body_ids(
    model: mujoco.MjModel,
    root_body_id: int,
) -> set[int]:
    descendants: set[int] = set()

    for body_id in range(model.nbody):
        current = body_id

        while current > 0:
            if current == root_body_id:
                descendants.add(body_id)
                break

            current = int(
                model.body_parentid[current]
            )

    descendants.add(root_body_id)

    return descendants


def actuator_id_for_joint(
    model: mujoco.MjModel,
    joint_id: int,
) -> int:
    matches = []

    for actuator_id in range(model.nu):
        transmitted_joint_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if transmitted_joint_id == joint_id:
            matches.append(actuator_id)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one actuator for joint id "
            f"{joint_id}, found {matches}."
        )

    return matches[0]


def arm_actuator_ids(
    model: mujoco.MjModel,
) -> list[int]:
    actuator_ids: list[int] = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise ValueError(
                f"Joint not found: {joint_name}"
            )

        actuator_ids.append(
            actuator_id_for_joint(
                model,
                joint_id,
            )
        )

    return actuator_ids


def extract_arm_qpos(
    model: mujoco.MjModel,
    full_qpos: np.ndarray,
) -> np.ndarray:
    values = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_address = int(
            model.jnt_qposadr[
                joint_id
            ]
        )

        values.append(
            float(
                full_qpos[
                    qpos_address
                ]
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def set_object_position(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    position: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        TARGET_BOX_FREEJOINT_NAME,
    )

    if joint_id < 0:
        raise ValueError(
            f"{TARGET_BOX_FREEJOINT_NAME} not found."
        )

    qpos_address = int(
        model.jnt_qposadr[
            joint_id
        ]
    )

    # Free joint qpos:
    # [x, y, z, qw, qx, qy, qz]
    qpos[
        qpos_address:
        qpos_address + 3
    ] = position


def longest_true_run(
    values: np.ndarray,
) -> int:
    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def contact_state(
    data: mujoco.MjData,
    *,
    object_geoms: set[int],
    moving_geoms: set[int],
    fixed_geoms: set[int],
) -> tuple[int, bool, bool]:
    count = 0
    moving_contact = False
    fixed_contact = False

    for contact_index in range(
        data.ncon
    ):
        contact = data.contact[
            contact_index
        ]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 in object_geoms:
            other = geom2
        elif geom2 in object_geoms:
            other = geom1
        else:
            continue

        if other in moving_geoms:
            moving_contact = True
            count += 1

        elif other in fixed_geoms:
            fixed_contact = True
            count += 1

    return (
        count,
        moving_contact,
        fixed_contact,
    )


def build_waypoint_trajectory(
    *,
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    object_position: np.ndarray,
    task: dict,
    pre_grasp_z_offset_m: float = 0.0,
):
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

    targets = tcp["targets"]

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
        model=model,
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

    qpos = initial_qpos.copy()

    home_arm_qpos = extract_arm_qpos(
        model,
        qpos,
    )

    waypoint_arm_qpos = {}

    # --------------------------------------------------------
    # 1. Pre-grasp
    # --------------------------------------------------------

    pre_phase = targets[
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

    pre_target[
        2
    ] += float(
        pre_grasp_z_offset_m
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
        raise RuntimeError(
            "IK failed for pre_grasp: "
            f"position error="
            f"{pre_result.position_error_mm:.3f} mm, "
            f"axis error="
            f"{pre_result.axis_error_deg:.3f} deg"
        )

    waypoint_arm_qpos[
        "pre_grasp"
    ] = extract_arm_qpos(
        model,
        pre_result.qpos,
    )

    qpos = (
        pre_result.qpos.copy()
    )

    # --------------------------------------------------------
    # Shared Lift waypoint definitions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. Reverse-Lift approach
    # --------------------------------------------------------

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
            raise RuntimeError(
                f"IK failed for "
                f"{waypoint_name}: "
                f"position error="
                f"{result.position_error_mm:.3f} mm, "
                f"axis error="
                f"{result.axis_error_deg:.3f} deg"
            )

        arm_qpos = extract_arm_qpos(
            model,
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

    # --------------------------------------------------------
    # 3. Grasp
    # --------------------------------------------------------

    grasp_phase = targets[
        "grasp"
    ]

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
        raise RuntimeError(
            "IK failed for grasp: "
            f"position error="
            f"{grasp_result.position_error_mm:.3f} mm, "
            f"axis error="
            f"{grasp_result.axis_error_deg:.3f} deg"
        )

    waypoint_arm_qpos[
        "grasp"
    ] = extract_arm_qpos(
        model,
        grasp_result.qpos,
    )

    qpos = (
        grasp_result.qpos.copy()
    )

    # --------------------------------------------------------
    # 4. Forward Lift-1 -> Lift-8
    # --------------------------------------------------------

    lift_waypoint_arm_qpos = []

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
            raise RuntimeError(
                f"IK failed for "
                f"{waypoint_name}: "
                f"position error="
                f"{result.position_error_mm:.3f} mm, "
                f"axis error="
                f"{result.axis_error_deg:.3f} deg"
            )

        arm_qpos = extract_arm_qpos(
            model,
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

    # Compatibility alias.
    waypoint_arm_qpos[
        "lift"
    ] = (
        lift_waypoint_arm_qpos[
            -1
        ]
    )

    return build_scripted_grasp_trajectory(
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


def run_one(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    task: dict,
    object_position: np.ndarray,
    object_offset: float,
    mapping_name: str,
    open_ctrl: float,
    closed_ctrl: float,
    arm_actuators: list[int],
    gripper_actuator: int,
    object_site_id: int,
    object_geoms: set[int],
    moving_geoms: set[int],
    fixed_geoms: set[int],
    home_joint_offsets_rad: np.ndarray | None = None,
    pre_grasp_z_offset_m: float = 0.0,
) -> MappingResult:
    mujoco.mj_resetData(
        model,
        data,
    )

    initial_qpos = (
        data.qpos.copy()
    )

    set_object_position(
        model,
        initial_qpos,
        object_position,
    )

    if home_joint_offsets_rad is not None:
        home_offsets = np.asarray(
            home_joint_offsets_rad,
            dtype=float,
        )

        if home_offsets.shape != (
            len(
                ARM_JOINT_NAMES
            ),
        ):
            raise ValueError(
                "home_joint_offsets_rad must have "
                f"shape ({len(ARM_JOINT_NAMES)},), "
                f"got {home_offsets.shape}."
            )

        if not np.all(
            np.isfinite(
                home_offsets
            )
        ):
            raise ValueError(
                "home_joint_offsets_rad contains "
                "non-finite values."
            )

        for (
            joint_name,
            offset_rad,
        ) in zip(
            ARM_JOINT_NAMES,
            home_offsets,
            strict=True,
        ):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:
                raise ValueError(
                    f"Arm joint not found: "
                    f"{joint_name}"
                )

            qpos_address = int(
                model.jnt_qposadr[
                    joint_id
                ]
            )

            initial_qpos[
                qpos_address
            ] += float(
                offset_rad
            )

    # Start in the open-gripper state.
    gripper_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        GRIPPER_JOINT_NAME,
    )

    gripper_qpos_address = int(
        model.jnt_qposadr[
            gripper_joint_id
        ]
    )

    initial_qpos[
        gripper_qpos_address
    ] = open_ctrl

    trajectory = (
        build_waypoint_trajectory(
            model=model,
            initial_qpos=initial_qpos,
            object_position=object_position,
            task=task,
            pre_grasp_z_offset_m=(
                pre_grasp_z_offset_m
            ),
        )
    )

    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0

    if model.na > 0:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    home_arm = (
        trajectory.arm_qpos[0]
    )

    for actuator_id, value in zip(
        arm_actuators,
        home_arm,
        strict=True,
    ):
        data.ctrl[
            actuator_id
        ] = value

    data.ctrl[
        gripper_actuator
    ] = open_ctrl

    mujoco.mj_forward(
        model,
        data,
    )

    initial_object_z = float(
        data.site_xpos[
            object_site_id,
            2,
        ]
    )

    object_z_by_frame = []
    moving_contact_by_frame = []
    fixed_contact_by_frame = []

    moving_contact_seen = False
    fixed_contact_seen = False
    two_side_contact_seen = False
    pre_close_contact_seen = False

    max_contact_count = 0

    physics_dt = float(
        model.opt.timestep
    )

    physics_step_count = 0

    close_start_frame = (
        phase_slices(
            trajectory.rate_hz
        )[
            "gripper_close"
        ].start
    )

    for frame_index in range(
        trajectory.frame_count
    ):
        arm_target = (
            trajectory.arm_qpos[
                frame_index
            ]
        )

        close_fraction = float(
            trajectory
            .gripper_close_fraction[
                frame_index
            ]
        )

        gripper_target = (
            open_ctrl
            + close_fraction
            * (
                closed_ctrl
                - open_ctrl
            )
        )

        for actuator_id, value in zip(
            arm_actuators,
            arm_target,
            strict=True,
        ):
            data.ctrl[
                actuator_id
            ] = value

        data.ctrl[
            gripper_actuator
        ] = gripper_target

        # Use cumulative time -> physics step conversion.
        # At 500 Hz physics and 30 Hz control this naturally
        # alternates 16/17 physics steps per trajectory frame.
        target_physics_steps = int(
            round(
                (
                    frame_index + 1
                )
                / trajectory.rate_hz
                / physics_dt
            )
        )

        frame_moving_contact = False
        frame_fixed_contact = False

        while (
            physics_step_count
            < target_physics_steps
        ):
            mujoco.mj_step(
                model,
                data,
            )

            physics_step_count += 1

            if not (
                np.all(
                    np.isfinite(
                        data.qpos
                    )
                )
                and np.all(
                    np.isfinite(
                        data.qvel
                    )
                )
            ):
                raise RuntimeError(
                    "Simulation became non-finite."
                )

            (
                contact_count,
                moving_contact,
                fixed_contact,
            ) = contact_state(
                data,
                object_geoms=object_geoms,
                moving_geoms=moving_geoms,
                fixed_geoms=fixed_geoms,
            )

            if (
                frame_index
                < close_start_frame
                and (
                    moving_contact
                    or fixed_contact
                )
            ):
                pre_close_contact_seen = True

            max_contact_count = max(
                max_contact_count,
                contact_count,
            )

            frame_moving_contact = (
                frame_moving_contact
                or moving_contact
            )

            frame_fixed_contact = (
                frame_fixed_contact
                or fixed_contact
            )

            moving_contact_seen = (
                moving_contact_seen
                or moving_contact
            )

            fixed_contact_seen = (
                fixed_contact_seen
                or fixed_contact
            )

            two_side_contact_seen = (
                two_side_contact_seen
                or (
                    moving_contact
                    and fixed_contact
                )
            )

        object_z_by_frame.append(
            float(
                data.site_xpos[
                    object_site_id,
                    2,
                ]
            )
        )

        moving_contact_by_frame.append(
            frame_moving_contact
        )

        fixed_contact_by_frame.append(
            frame_fixed_contact
        )

    object_z = np.asarray(
        object_z_by_frame,
        dtype=float,
    )

    lift = (
        object_z
        - initial_object_z
    )

    success_config = task[
        "success"
    ]

    minimum_lift = float(
        success_config[
            "minimum_lift_delta_m"
        ]
    )

    hold_duration_s = float(
        success_config[
            "hold_duration_s"
        ]
    )

    required_hold_frames = int(
        math.ceil(
            hold_duration_s
            * trajectory.rate_hz
            - 1e-12
        )
    )

    final_hold_slice = (
        phase_slices(
            trajectory.rate_hz
        )[
            "final_hold"
        ]
    )

    final_hold_above_threshold = (
        lift[
            final_hold_slice
        ]
        >= minimum_lift
    )

    hold_frames = longest_true_run(
        final_hold_above_threshold
    )

    success = (
        hold_frames
        >= required_hold_frames
    )

    return MappingResult(
        mapping_name=mapping_name,
        offset_m=float(
            object_offset
        ),
        success=success,
        initial_object_z=initial_object_z,
        final_object_z=float(
            object_z[-1]
        ),
        max_object_z=float(
            np.max(object_z)
        ),
        final_lift_m=float(
            lift[-1]
        ),
        max_lift_m=float(
            np.max(lift)
        ),
        hold_frames=hold_frames,
        required_hold_frames=required_hold_frames,
        moving_jaw_contact_seen=moving_contact_seen,
        fixed_finger_contact_seen=fixed_contact_seen,
        simultaneous_two_side_contact_seen=two_side_contact_seen,
        pre_close_contact_seen=pre_close_contact_seen,
        max_gripper_object_contacts=max_contact_count,
        simulation_time_s=float(
            data.time
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiment/"
            "response_comparison.yaml"
        ),
    )

    parser.add_argument(
        "--scene-config",
        default=(
            "configs/scene/"
            "top_grasp.yaml"
        ),
    )

    parser.add_argument(
        "--task-config",
        default=(
            "configs/task/"
            "pick_object.yaml"
        ),
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    args = parser.parse_args()

    try:
        bundle = load_experiment_bundle(
            Path(
                args.experiment_config
            )
        )

        env = (
            MujocoEnvironment
            .from_robot_config(
                bundle.section(
                    "robot"
                ).root,
                repo_root=Path(
                    args.repo_root
                ),
            )
        )

        env.reset()

    except (
        ConfigError,
        SimulationError,
    ) as exc:
        print(
            f"Physics check failed: {exc}"
        )
        return 1

    model = env.model
    data = env.data

    scene = load_yaml(
        Path(
            args.scene_config
        )
    )["scene"]

    task = load_yaml(
        Path(
            args.task_config
        )
    )["task"]

    perturbation = scene[
        "perturbation"
    ]

    perturb_axis = perturbation[
        "axis"
    ]

    perturb_index = AXIS_TO_INDEX[
        perturb_axis
    ]

    training_offsets = perturbation[
        "training_offsets_m"
    ]

    baseline_position = np.asarray(
        scene["object"][
            "baseline_pose"
        ]["position_m"],
        dtype=float,
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

    if object_site_id < 0:
        print(
            "Physics check failed: "
            "target_box_center not found."
        )
        return 1

    target_box_body_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            TARGET_BOX_BODY_NAME,
        )
    )

    gripper_link_body_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            GRIPPER_LINK_BODY_NAME,
        )
    )

    jaw_link_body_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            JAW_LINK_BODY_NAME,
        )
    )

    if min(
        target_box_body_id,
        gripper_link_body_id,
        jaw_link_body_id,
    ) < 0:
        print(
            "Physics check failed: "
            "required body not found."
        )
        return 1

    object_bodies = descendant_body_ids(
        model,
        target_box_body_id,
    )

    jaw_bodies = descendant_body_ids(
        model,
        jaw_link_body_id,
    )

    gripper_bodies = descendant_body_ids(
        model,
        gripper_link_body_id,
    )

    fixed_gripper_bodies = (
        gripper_bodies
        - jaw_bodies
    )

    object_geoms = object_geom_ids(
        model,
        object_bodies,
    )

    moving_geoms = object_geom_ids(
        model,
        jaw_bodies,
    )

    fixed_geoms = object_geom_ids(
        model,
        fixed_gripper_bodies,
    )

    arm_actuators = (
        arm_actuator_ids(
            model
        )
    )

    gripper_joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        GRIPPER_JOINT_NAME,
    )

    gripper_actuator = (
        actuator_id_for_joint(
            model,
            gripper_joint_id,
        )
    )

    mappings = (
        (
            "candidate_A_upper_open",
            1.0,
            0.0,
        ),
        (
            "candidate_B_lower_open",
            0.0,
            1.0,
        ),
    )

    print(
        "SO-101 MuJoCo grasp physics check"
    )
    print()
    print(
        f"physics timestep: "
        f"{model.opt.timestep:.6f} s"
    )
    print(
        f"physics rate: "
        f"{1.0 / model.opt.timestep:.1f} Hz"
    )
    print(
        f"trajectory rate: "
        f"{DEFAULT_RATE_HZ:.1f} Hz"
    )
    print(
        f"object geoms: "
        f"{sorted(object_geoms)}"
    )
    print(
        f"moving jaw geoms: "
        f"{sorted(moving_geoms)}"
    )
    print(
        f"fixed gripper geoms: "
        f"{sorted(fixed_geoms)}"
    )

    all_results = []

    for (
        mapping_name,
        open_ctrl,
        closed_ctrl,
    ) in mappings:
        print()
        print("#" * 76)
        print(
            f"Mapping: {mapping_name}"
        )
        print(
            f"  open ctrl   = "
            f"{open_ctrl:.1f}"
        )
        print(
            f"  closed ctrl = "
            f"{closed_ctrl:.1f}"
        )

        for object_offset in (
            training_offsets
        ):
            object_position = (
                baseline_position.copy()
            )

            object_position[
                perturb_index
            ] += float(
                object_offset
            )

            result = run_one(
                model=model,
                data=data,
                task=task,
                object_position=object_position,
                object_offset=float(
                    object_offset
                ),
                mapping_name=mapping_name,
                open_ctrl=open_ctrl,
                closed_ctrl=closed_ctrl,
                arm_actuators=arm_actuators,
                gripper_actuator=gripper_actuator,
                object_site_id=object_site_id,
                object_geoms=object_geoms,
                moving_geoms=moving_geoms,
                fixed_geoms=fixed_geoms,
            )

            all_results.append(
                result
            )

            print()
            print(
                "=" * 72
            )
            print(
                "Object offset:",
                f"{result.offset_m * 1000:+.1f} mm",
            )
            print(
                "  success: "
                f"{result.success}"
            )
            print(
                "  initial object z: "
                f"{result.initial_object_z:.6f} m"
            )
            print(
                "  final object z:   "
                f"{result.final_object_z:.6f} m"
            )
            print(
                "  max object z:     "
                f"{result.max_object_z:.6f} m"
            )
            print(
                "  final lift:        "
                f"{result.final_lift_m * 1000:.2f} mm"
            )
            print(
                "  max lift:          "
                f"{result.max_lift_m * 1000:.2f} mm"
            )
            print(
                "  hold above 70 mm:  "
                f"{result.hold_frames}/"
                f"{result.required_hold_frames} frames"
            )
            print(
                "  moving jaw contact:"
                f" {result.moving_jaw_contact_seen}"
            )
            print(
                "  fixed side contact:"
                f" {result.fixed_finger_contact_seen}"
            )
            print(
                "  simultaneous two-side contact:"
                f" {result.simultaneous_two_side_contact_seen}"
            )
            print(
                "  max object/gripper contacts:"
                f" {result.max_gripper_object_contacts}"
            )
            print(
                "  simulation time:"
                f" {result.simulation_time_s:.4f} s"
            )

    print()
    print("#" * 76)
    print(
        "Mapping summary"
    )

    mapping_success_counts = {}

    for mapping_name, _, _ in mappings:
        matching = [
            result
            for result in all_results
            if (
                result.mapping_name
                == mapping_name
            )
        ]

        success_count = sum(
            int(result.success)
            for result in matching
        )

        mapping_success_counts[
            mapping_name
        ] = success_count

        print(
            f"  {mapping_name:<28}"
            f"{success_count}/"
            f"{len(matching)} successes"
        )

    winner = [
        mapping_name
        for mapping_name, count
        in mapping_success_counts.items()
        if count == len(
            training_offsets
        )
    ]

    print()

    if len(winner) == 1:
        print(
            "GRIPPER MAPPING CANDIDATE CONFIRMED:"
        )
        print(
            f"  {winner[0]}"
        )
    elif len(winner) > 1:
        print(
            "MAPPING INCONCLUSIVE:"
            " both mappings satisfied the lift criterion."
        )
    else:
        print(
            "GRASP PHYSICS NOT YET VALIDATED:"
            " neither mapping succeeded at all training positions."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
