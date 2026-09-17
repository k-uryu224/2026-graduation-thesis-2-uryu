import numpy as np

from so101_vla_response.trajectory.scripted_grasp import (
    allocate_corridor_frames,
    build_scripted_grasp_trajectory,
    phase_frame_counts,
    phase_slices,
)


def arm(value):
    return np.full(
        5,
        float(value),
        dtype=float,
    )


def test_phase_counts_are_210_frames():
    counts = phase_frame_counts(
        30.0
    )

    assert counts == {
        "initial_hold": 15,
        "home_to_pre_grasp": 60,
        "pre_grasp_to_grasp": 45,
        "gripper_close": 15,
        "grasp_to_lift": 45,
        "final_hold": 30,
    }

    assert sum(
        counts.values()
    ) == 210


def test_legacy_builder_is_backward_compatible():
    home = arm(0.0)
    pre = arm(0.1)
    grasp = arm(0.2)
    lift = arm(0.3)

    trajectory = (
        build_scripted_grasp_trajectory(
            home_arm_qpos=home,
            pre_grasp_arm_qpos=pre,
            grasp_arm_qpos=grasp,
            lift_arm_qpos=lift,
            rate_hz=30.0,
        )
    )

    assert (
        trajectory.arm_qpos.shape
        == (210, 5)
    )

    assert (
        trajectory.gripper_close_fraction.shape
        == (210,)
    )

    assert trajectory.frame_count == 210

    assert np.isclose(
        trajectory.duration_s,
        7.0,
    )

    slices = phase_slices(
        30.0
    )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["initial_hold"]
        ],
        home,
    )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["home_to_pre_grasp"].stop
            - 1
        ],
        pre,
    )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["pre_grasp_to_grasp"].stop
            - 1
        ],
        grasp,
    )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["grasp_to_lift"].stop
            - 1
        ],
        lift,
    )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["final_hold"]
        ],
        lift,
    )


def test_corridor_frame_allocation():
    points = [
        arm(0.00),
        arm(0.05),
        arm(0.10),
        arm(0.30),
    ]

    allocation = (
        allocate_corridor_frames(
            points,
            45,
        )
    )

    assert len(
        allocation
    ) == 3

    assert sum(
        allocation
    ) == 45

    assert all(
        count >= 3
        for count in allocation
    )

    # Last segment has the largest joint change,
    # therefore it should receive the most frames.
    assert (
        allocation[-1]
        > allocation[0]
    )


def test_corridor_builder_visits_all_waypoints():
    home = arm(0.0)
    pre = arm(0.1)

    approach_waypoints = [
        arm(0.12),
        arm(0.14),
        arm(0.16),
        arm(0.18),
        arm(0.20),
        arm(0.22),
    ]

    grasp = arm(0.24)

    lift_waypoints = [
        arm(0.26),
        arm(0.28),
        arm(0.30),
        arm(0.32),
        arm(0.34),
        arm(0.36),
        arm(0.38),
        arm(0.40),
    ]

    trajectory = (
        build_scripted_grasp_trajectory(
            home_arm_qpos=home,
            pre_grasp_arm_qpos=pre,
            grasp_arm_qpos=grasp,
            approach_waypoint_arm_qpos=(
                approach_waypoints
            ),
            lift_waypoint_arm_qpos=(
                lift_waypoints
            ),
            rate_hz=30.0,
        )
    )

    assert trajectory.frame_count == 210

    slices = phase_slices(
        30.0
    )

    approach_corridor = [
        pre,
        *approach_waypoints,
        grasp,
    ]

    approach_allocation = (
        allocate_corridor_frames(
            approach_corridor,
            45,
        )
    )

    start = (
        slices[
            "pre_grasp_to_grasp"
        ].start
    )

    cumulative = 0

    for (
        waypoint,
        count,
    ) in zip(
        approach_corridor[
            1:
        ],
        approach_allocation,
        strict=True,
    ):
        cumulative += count

        endpoint = (
            start
            + cumulative
            - 1
        )

        assert np.allclose(
            trajectory.arm_qpos[
                endpoint
            ],
            waypoint,
        )

    lift_corridor = [
        grasp,
        *lift_waypoints,
    ]

    lift_allocation = (
        allocate_corridor_frames(
            lift_corridor,
            45,
        )
    )

    start = (
        slices[
            "grasp_to_lift"
        ].start
    )

    cumulative = 0

    for (
        waypoint,
        count,
    ) in zip(
        lift_corridor[
            1:
        ],
        lift_allocation,
        strict=True,
    ):
        cumulative += count

        endpoint = (
            start
            + cumulative
            - 1
        )

        assert np.allclose(
            trajectory.arm_qpos[
                endpoint
            ],
            waypoint,
        )

    assert np.allclose(
        trajectory.arm_qpos[
            slices["final_hold"]
        ],
        lift_waypoints[-1],
    )


def test_corridor_keeps_gripper_open_until_close():
    trajectory = (
        build_scripted_grasp_trajectory(
            home_arm_qpos=arm(0.0),
            pre_grasp_arm_qpos=arm(0.1),
            grasp_arm_qpos=arm(0.2),
            approach_waypoint_arm_qpos=[
                arm(0.12),
                arm(0.15),
            ],
            lift_waypoint_arm_qpos=[
                arm(0.25),
                arm(0.30),
            ],
            rate_hz=30.0,
        )
    )

    slices = phase_slices(
        30.0
    )

    assert np.allclose(
        trajectory.gripper_close_fraction[
            :slices["gripper_close"].start
        ],
        0.0,
    )

    close_values = (
        trajectory.gripper_close_fraction[
            slices["gripper_close"]
        ]
    )

    assert np.all(
        close_values > 0.0
    )

    assert np.isclose(
        close_values[-1],
        1.0,
    )

    assert np.allclose(
        trajectory.gripper_close_fraction[
            slices["grasp_to_lift"]
        ],
        1.0,
    )

    assert np.allclose(
        trajectory.gripper_close_fraction[
            slices["final_hold"]
        ],
        1.0,
    )
