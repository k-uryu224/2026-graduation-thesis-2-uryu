"""Deterministic scripted grasp trajectory generation for SO-101."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


DEFAULT_RATE_HZ = 30.0
ARM_DOF_COUNT = 5

PHASE_DURATIONS_S = (
    ("initial_hold", 0.5),
    ("home_to_pre_grasp", 2.0),
    ("pre_grasp_to_grasp", 1.5),
    ("gripper_close", 0.5),
    ("grasp_to_lift", 1.5),
    ("final_hold", 1.0),
)


def smoothstep(tau):
    """Cubic smoothstep s(tau)=3*tau^2-2*tau^3 for tau in [0, 1]."""

    values = np.asarray(
        tau,
        dtype=float,
    )

    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            "smoothstep input must be inside [0, 1]."
        )

    return (
        3.0 * values**2
        - 2.0 * values**3
    )


def phase_frame_counts(
    rate_hz: float = DEFAULT_RATE_HZ,
) -> dict[str, int]:
    """Return the exact number of samples assigned to each phase."""

    if rate_hz <= 0.0:
        raise ValueError(
            "rate_hz must be positive."
        )

    counts: dict[str, int] = {}

    for (
        phase_name,
        duration_s,
    ) in PHASE_DURATIONS_S:
        raw_count = (
            duration_s
            * rate_hz
        )

        frame_count = int(
            round(
                raw_count
            )
        )

        if not math.isclose(
            raw_count,
            frame_count,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Phase {phase_name!r} does not map to "
                f"an integer number of frames at "
                f"{rate_hz} Hz."
            )

        counts[
            phase_name
        ] = frame_count

    return counts


def phase_slices(
    rate_hz: float = DEFAULT_RATE_HZ,
) -> dict[str, slice]:
    """Return frame slices for each trajectory phase."""

    slices: dict[
        str,
        slice,
    ] = {}

    start = 0

    for (
        phase_name,
        count,
    ) in phase_frame_counts(
        rate_hz
    ).items():
        stop = (
            start
            + count
        )

        slices[
            phase_name
        ] = slice(
            start,
            stop,
        )

        start = stop

    return slices


def _as_arm_vector(
    values,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        values,
        dtype=float,
    )

    if vector.shape != (
        ARM_DOF_COUNT,
    ):
        raise ValueError(
            f"{name} must have shape "
            f"({ARM_DOF_COUNT},), "
            f"got {vector.shape}."
        )

    if not np.all(
        np.isfinite(
            vector
        )
    ):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return vector.copy()


def _as_waypoint_sequence(
    values,
    name: str,
) -> list[np.ndarray]:
    if values is None:
        return []

    waypoints = []

    for (
        index,
        value,
    ) in enumerate(
        values
    ):
        waypoints.append(
            _as_arm_vector(
                value,
                f"{name}[{index}]",
            )
        )

    return waypoints


def _joint_segment(
    start: np.ndarray,
    goal: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Generate a smoothstep segment ending exactly at goal."""

    if frame_count <= 0:
        raise ValueError(
            "frame_count must be positive."
        )

    # The previous phase already contains the exact
    # start configuration. Therefore sample
    # tau=(1/N)...1 rather than tau=0...1.
    tau = (
        np.arange(
            1,
            frame_count + 1,
            dtype=float,
        )
        / float(
            frame_count
        )
    )

    blend = smoothstep(
        tau
    )[:, None]

    return (
        start[None, :]
        + blend
        * (
            goal
            - start
        )[None, :]
    )


def allocate_corridor_frames(
    waypoints,
    total_frames: int,
    *,
    minimum_frames_per_segment: int = 3,
) -> list[int]:
    """Allocate corridor frames by maximum joint change.

    This is the frame-allocation rule validated during the
    calibrated SO-101 grasp physics sweep:

    - each corridor segment receives at least three frames;
    - remaining frames are distributed in proportion to the
      maximum absolute arm-joint change of each segment.
    """

    points = _as_waypoint_sequence(
        waypoints,
        "waypoints",
    )

    if len(points) < 2:
        raise ValueError(
            "At least two waypoints are required."
        )

    if total_frames <= 0:
        raise ValueError(
            "total_frames must be positive."
        )

    if minimum_frames_per_segment <= 0:
        raise ValueError(
            "minimum_frames_per_segment "
            "must be positive."
        )

    weights = []

    for (
        start,
        goal,
    ) in zip(
        points[:-1],
        points[1:],
        strict=True,
    ):
        weights.append(
            max(
                float(
                    np.max(
                        np.abs(
                            goal
                            - start
                        )
                    )
                ),
                1e-6,
            )
        )

    segment_count = len(
        weights
    )

    base = (
        minimum_frames_per_segment
    )

    if (
        base
        * segment_count
        > total_frames
    ):
        raise ValueError(
            "Not enough frames for the corridor."
        )

    allocation = np.full(
        segment_count,
        base,
        dtype=int,
    )

    remaining = (
        total_frames
        - base
        * segment_count
    )

    weights_array = np.asarray(
        weights,
        dtype=float,
    )

    ideal = (
        remaining
        * weights_array
        / np.sum(
            weights_array
        )
    )

    extra = np.floor(
        ideal
    ).astype(
        int
    )

    allocation += extra

    leftover = (
        total_frames
        - int(
            np.sum(
                allocation
            )
        )
    )

    fractions = (
        ideal
        - extra
    )

    order = np.argsort(
        -fractions
    )

    for index in order[
        :leftover
    ]:
        allocation[
            index
        ] += 1

    if (
        int(
            np.sum(
                allocation
            )
        )
        != total_frames
    ):
        raise RuntimeError(
            "Frame allocation error."
        )

    return allocation.tolist()


def _corridor_segment(
    waypoints,
    total_frames: int,
) -> np.ndarray:
    points = _as_waypoint_sequence(
        waypoints,
        "corridor_waypoints",
    )

    allocation = (
        allocate_corridor_frames(
            points,
            total_frames,
        )
    )

    segments = []

    for (
        start,
        goal,
        count,
    ) in zip(
        points[:-1],
        points[1:],
        allocation,
        strict=True,
    ):
        segments.append(
            _joint_segment(
                start,
                goal,
                count,
            )
        )

    return np.concatenate(
        segments,
        axis=0,
    )


@dataclass(frozen=True)
class ScriptedGraspTrajectory:
    """A complete fixed-rate grasp trajectory."""

    arm_qpos: np.ndarray
    gripper_close_fraction: np.ndarray
    phases: tuple[str, ...]
    timestamps_s: np.ndarray
    rate_hz: float

    @property
    def frame_count(
        self,
    ) -> int:
        return int(
            self.arm_qpos.shape[
                0
            ]
        )

    @property
    def duration_s(
        self,
    ) -> float:
        return (
            self.frame_count
            / self.rate_hz
        )

    def phase_counts(
        self,
    ) -> dict[str, int]:
        return dict(
            Counter(
                self.phases
            )
        )


def build_scripted_grasp_trajectory(
    *,
    home_arm_qpos,
    pre_grasp_arm_qpos,
    grasp_arm_qpos,
    lift_arm_qpos=None,
    approach_waypoint_arm_qpos=None,
    lift_waypoint_arm_qpos=None,
    rate_hz: float = DEFAULT_RATE_HZ,
) -> ScriptedGraspTrajectory:
    """Build a fixed-rate SO-101 scripted grasp trajectory.

    Legacy mode
    -----------
    If no intermediate waypoint arrays are supplied, this
    preserves the original behavior:

        Home -> Pre -> Grasp -> Close -> Lift -> Hold

    Corridor mode
    -------------
    approach_waypoint_arm_qpos contains intermediate arm
    configurations between Pre and Grasp.

    lift_waypoint_arm_qpos contains the ordered Lift
    configurations after Grasp.

    The calibrated experiment uses:

        Pre
        -> reverse Lift-6
        -> reverse Lift-5
        -> ...
        -> reverse Lift-1
        -> Grasp
        -> Close
        -> Lift-1
        -> ...
        -> Lift-8
        -> Hold

    The gripper signal is represented as normalized close
    fraction:

        0.0 = open
        1.0 = closed
    """

    home = _as_arm_vector(
        home_arm_qpos,
        "home_arm_qpos",
    )

    pre_grasp = _as_arm_vector(
        pre_grasp_arm_qpos,
        "pre_grasp_arm_qpos",
    )

    grasp = _as_arm_vector(
        grasp_arm_qpos,
        "grasp_arm_qpos",
    )

    approach_waypoints = (
        _as_waypoint_sequence(
            approach_waypoint_arm_qpos,
            "approach_waypoint_arm_qpos",
        )
    )

    lift_waypoints = (
        _as_waypoint_sequence(
            lift_waypoint_arm_qpos,
            "lift_waypoint_arm_qpos",
        )
    )

    legacy_lift = None

    if lift_arm_qpos is not None:
        legacy_lift = (
            _as_arm_vector(
                lift_arm_qpos,
                "lift_arm_qpos",
            )
        )

    if lift_waypoints:
        final_lift = (
            lift_waypoints[
                -1
            ]
        )

        if (
            legacy_lift
            is not None
            and not np.allclose(
                legacy_lift,
                final_lift,
                atol=1e-9,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "lift_arm_qpos must match the final "
                "lift waypoint when both are supplied."
            )

    else:
        if legacy_lift is None:
            raise ValueError(
                "Either lift_arm_qpos or "
                "lift_waypoint_arm_qpos is required."
            )

        final_lift = (
            legacy_lift
        )

    counts = phase_frame_counts(
        rate_hz
    )

    arm_segments: list[
        np.ndarray
    ] = []

    gripper_segments: list[
        np.ndarray
    ] = []

    phases: list[
        str
    ] = []

    # --------------------------------------------------------
    # 1. Initial hold
    # --------------------------------------------------------

    count = counts[
        "initial_hold"
    ]

    arm_segments.append(
        np.repeat(
            home[None, :],
            count,
            axis=0,
        )
    )

    gripper_segments.append(
        np.zeros(
            count,
            dtype=float,
        )
    )

    phases.extend(
        [
            "initial_hold"
        ]
        * count
    )

    # --------------------------------------------------------
    # 2. Home -> Pre-grasp
    # --------------------------------------------------------

    count = counts[
        "home_to_pre_grasp"
    ]

    arm_segments.append(
        _joint_segment(
            home,
            pre_grasp,
            count,
        )
    )

    gripper_segments.append(
        np.zeros(
            count,
            dtype=float,
        )
    )

    phases.extend(
        [
            "home_to_pre_grasp"
        ]
        * count
    )

    # --------------------------------------------------------
    # 3. Pre-grasp -> Grasp
    # --------------------------------------------------------

    count = counts[
        "pre_grasp_to_grasp"
    ]

    approach_corridor = [
        pre_grasp,
        *approach_waypoints,
        grasp,
    ]

    arm_segments.append(
        _corridor_segment(
            approach_corridor,
            count,
        )
    )

    gripper_segments.append(
        np.zeros(
            count,
            dtype=float,
        )
    )

    phases.extend(
        [
            "pre_grasp_to_grasp"
        ]
        * count
    )

    # --------------------------------------------------------
    # 4. Close gripper while holding Grasp
    # --------------------------------------------------------

    count = counts[
        "gripper_close"
    ]

    arm_segments.append(
        np.repeat(
            grasp[None, :],
            count,
            axis=0,
        )
    )

    close_tau = (
        np.arange(
            1,
            count + 1,
            dtype=float,
        )
        / float(
            count
        )
    )

    gripper_segments.append(
        smoothstep(
            close_tau
        )
    )

    phases.extend(
        [
            "gripper_close"
        ]
        * count
    )

    # --------------------------------------------------------
    # 5. Grasp -> Lift corridor
    # --------------------------------------------------------

    count = counts[
        "grasp_to_lift"
    ]

    if lift_waypoints:
        lift_corridor = [
            grasp,
            *lift_waypoints,
        ]

    else:
        lift_corridor = [
            grasp,
            final_lift,
        ]

    arm_segments.append(
        _corridor_segment(
            lift_corridor,
            count,
        )
    )

    gripper_segments.append(
        np.ones(
            count,
            dtype=float,
        )
    )

    phases.extend(
        [
            "grasp_to_lift"
        ]
        * count
    )

    # --------------------------------------------------------
    # 6. Final hold
    # --------------------------------------------------------

    count = counts[
        "final_hold"
    ]

    arm_segments.append(
        np.repeat(
            final_lift[
                None,
                :
            ],
            count,
            axis=0,
        )
    )

    gripper_segments.append(
        np.ones(
            count,
            dtype=float,
        )
    )

    phases.extend(
        [
            "final_hold"
        ]
        * count
    )

    arm_qpos = np.concatenate(
        arm_segments,
        axis=0,
    )

    gripper_close_fraction = (
        np.concatenate(
            gripper_segments,
            axis=0,
        )
    )

    frame_count = (
        arm_qpos.shape[
            0
        ]
    )

    timestamps_s = (
        np.arange(
            frame_count,
            dtype=float,
        )
        / float(
            rate_hz
        )
    )

    return ScriptedGraspTrajectory(
        arm_qpos=arm_qpos,
        gripper_close_fraction=(
            gripper_close_fraction
        ),
        phases=tuple(
            phases
        ),
        timestamps_s=timestamps_s,
        rate_hz=float(
            rate_hz
        ),
    )
