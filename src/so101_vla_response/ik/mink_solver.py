"""Mink-based inverse kinematics for the SO-101 arm."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mink
import mujoco
import numpy as np


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)


@dataclass(frozen=True)
class IKResult:
    """Result of one SO-101 IK solve."""

    reached: bool
    iterations: int
    qpos: np.ndarray
    position_error_m: float
    axis_error_rad: float
    min_joint_margin_rad: float

    @property
    def position_error_mm(self) -> float:
        return self.position_error_m * 1000.0

    @property
    def axis_error_deg(self) -> float:
        return math.degrees(self.axis_error_rad)

    @property
    def min_joint_margin_deg(self) -> float:
        return math.degrees(self.min_joint_margin_rad)


class SO101MinkIK:
    """Differential IK using only the five SO-101 arm DOFs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        tcp_site: str = "gripperframe",
        local_approach_axis=(1.0, 0.0, 0.0),
        local_pinch_axis=(0.0, 0.0, 1.0),
        dt: float = 0.01,
        max_iterations: int = 500,
        position_tolerance_m: float = 0.002,
        axis_tolerance_rad: float = math.radians(2.0),
        posture_cost: float = 1e-3,
        pinch_cost: float = 0.05,
    ) -> None:
        self.model = model
        self.tcp_site = tcp_site
        self.local_approach_axis = self._unit(
            local_approach_axis
        )
        self.local_pinch_axis = self._unit(
            local_pinch_axis
        )

        if abs(
            float(
                np.dot(
                    self.local_approach_axis,
                    self.local_pinch_axis,
                )
            )
        ) > 1e-6:
            raise ValueError(
                "local_approach_axis and local_pinch_axis "
                "must be orthogonal."
            )

        self.dt = dt
        self.max_iterations = max_iterations
        self.position_tolerance_m = position_tolerance_m
        self.axis_tolerance_rad = axis_tolerance_rad
        self.posture_cost = posture_cost
        self.pinch_cost = float(pinch_cost)

        if self.pinch_cost < 0.0:
            raise ValueError(
                "pinch_cost must be non-negative."
            )

        self.arm_dof_indices = self._find_arm_dofs()

        arm_dofs = set(self.arm_dof_indices)

        self.frozen_dof_indices = tuple(
            index
            for index in range(model.nv)
            if index not in arm_dofs
        )

    @staticmethod
    def _unit(vector) -> np.ndarray:
        array = np.asarray(vector, dtype=float)

        if array.shape != (3,):
            raise ValueError(
                "Direction vector must have shape (3,)."
            )

        norm = float(np.linalg.norm(array))

        if norm <= 0.0:
            raise ValueError(
                "Direction vector must be non-zero."
            )

        return array / norm

    def _find_arm_dofs(self) -> tuple[int, ...]:
        indices: list[int] = []

        for joint_name in ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:
                raise ValueError(
                    f"Required joint not found: {joint_name}"
                )

            indices.append(
                int(self.model.jnt_dofadr[joint_id])
            )

        return tuple(indices)

    def _errors(
        self,
        configuration: mink.Configuration,
        target_position: np.ndarray,
        target_direction: np.ndarray,
    ) -> tuple[float, float]:
        transform = (
            configuration
            .get_transform_frame_to_world(
                self.tcp_site,
                "site",
            )
            .as_matrix()
        )

        position = transform[:3, 3]
        rotation = transform[:3, :3]

        position_error = float(
            np.linalg.norm(
                position - target_position
            )
        )

        actual_axis = (
            rotation @ self.local_approach_axis
        )
        actual_axis = self._unit(actual_axis)

        dot = float(
            np.clip(
                np.dot(
                    actual_axis,
                    target_direction,
                ),
                -1.0,
                1.0,
            )
        )

        axis_error = float(
            math.acos(dot)
        )

        return position_error, axis_error

    def _minimum_arm_joint_margin(
        self,
        qpos: np.ndarray,
    ) -> float:
        margins: list[float] = []

        for joint_name in ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            qpos_address = int(
                self.model.jnt_qposadr[joint_id]
            )

            value = float(
                qpos[qpos_address]
            )

            lower = float(
                self.model.jnt_range[joint_id][0]
            )
            upper = float(
                self.model.jnt_range[joint_id][1]
            )

            margins.append(
                min(
                    value - lower,
                    upper - value,
                )
            )

        return min(margins)

    def solve(
        self,
        initial_qpos,
        target_position,
        target_direction,
        *,
        target_pinch_direction=None,
    ) -> IKResult:
        """Solve one TCP position + approach-direction target.

        ``target_pinch_direction`` is optional. When supplied,
        gripper local Z is weakly aligned with that world-space
        direction to reduce otherwise free roll about the
        approach axis.
        """

        target_position = np.asarray(
            target_position,
            dtype=float,
        )

        if target_position.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,)."
            )

        target_direction = self._unit(
            target_direction
        )

        if target_pinch_direction is not None:
            target_pinch_direction = self._unit(
                target_pinch_direction
            )

            orthogonality = abs(
                float(
                    np.dot(
                        target_direction,
                        target_pinch_direction,
                    )
                )
            )

            if orthogonality > 1e-5:
                raise ValueError(
                    "target_direction and "
                    "target_pinch_direction must be orthogonal."
                )

        configuration = mink.Configuration(
            self.model
        )

        configuration.update(
            np.asarray(
                initial_qpos,
                dtype=float,
            ).copy()
        )

        position_task = mink.FrameTask(
            frame_name=self.tcp_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            gain=0.5,
            lm_damping=1e-4,
        )

        position_task.set_target(
            mink.SE3.from_translation(
                target_position
            )
        )

        axis_task = mink.AxisAlignTask(
            frame_name=self.tcp_site,
            frame_type="site",
            axis=self.local_approach_axis,
            cost=1.0,
            gain=0.5,
            lm_damping=1e-4,
        )

        axis_task.set_target(
            target_direction
        )

        pinch_task = None

        if (
            target_pinch_direction is not None
            and self.pinch_cost > 0.0
        ):
            pinch_task = mink.AxisAlignTask(
                frame_name=self.tcp_site,
                frame_type="site",
                axis=self.local_pinch_axis,
                cost=self.pinch_cost,
                gain=0.5,
                lm_damping=1e-4,
            )

            pinch_task.set_target(
                target_pinch_direction
            )

        posture_task = mink.PostureTask(
            model=self.model,
            cost=self.posture_cost,
            gain=1.0,
            lm_damping=1e-4,
        )

        posture_task.set_target_from_configuration(
            configuration
        )

        freeze_task = mink.DofFreezingTask(
            model=self.model,
            dof_indices=self.frozen_dof_indices,
        )

        limits = [
            mink.ConfigurationLimit(
                model=self.model
            )
        ]

        tasks = [
            position_task,
            axis_task,
        ]

        if pinch_task is not None:
            tasks.append(
                pinch_task
            )

        tasks.append(
            posture_task
        )

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            (
                position_error,
                axis_error,
            ) = self._errors(
                configuration,
                target_position,
                target_direction,
            )

            joint_margin = (
                self._minimum_arm_joint_margin(
                    configuration.q
                )
            )

            if (
                position_error
                <= self.position_tolerance_m
                and axis_error
                <= self.axis_tolerance_rad
                and joint_margin >= 0.0
            ):
                return IKResult(
                    reached=True,
                    iterations=iteration - 1,
                    qpos=configuration.q.copy(),
                    position_error_m=position_error,
                    axis_error_rad=axis_error,
                    min_joint_margin_rad=joint_margin,
                )

            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                dt=self.dt,
                solver="daqp",
                damping=1e-8,
                safety_break=True,
                limits=limits,
                constraints=[freeze_task],
            )

            configuration.integrate_inplace(
                velocity,
                self.dt,
            )

        (
            position_error,
            axis_error,
        ) = self._errors(
            configuration,
            target_position,
            target_direction,
        )

        joint_margin = (
            self._minimum_arm_joint_margin(
                configuration.q
            )
        )

        return IKResult(
            reached=False,
            iterations=self.max_iterations,
            qpos=configuration.q.copy(),
            position_error_m=position_error,
            axis_error_rad=axis_error,
            min_joint_margin_rad=joint_margin,
        )
