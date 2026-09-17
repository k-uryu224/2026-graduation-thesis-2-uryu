"""Thin MuJoCo environment wrapper.

This module owns only model loading, reset, control assignment, and stepping.
Scene construction, IK targets, trajectory phases, dataset recording, and policy
adapters live in separate modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SimulationError(RuntimeError):
    """Raised when the simulation environment cannot be used safely."""


class MujocoImportError(SimulationError):
    """Raised when MuJoCo is required but not installed."""


@dataclass(frozen=True)
class MujocoModelSummary:
    """Basic model metadata useful for asset verification."""

    xml_path: Path
    nq: int
    nv: int
    nu: int
    nbody: int
    ngeom: int
    nsite: int
    ncam: int
    timestep_s: float
    joint_names: tuple[str, ...]
    site_names: tuple[str, ...]
    camera_names: tuple[str, ...]


@dataclass(frozen=True)
class SimulationState:
    """Snapshot of MuJoCo state after reset or step."""

    time_s: float
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ctrl: tuple[float, ...]


class MujocoEnvironment:
    """Load, reset, and step a MuJoCo model."""

    def __init__(self, model_xml_path: str | Path, *, mujoco_module: Any | None = None) -> None:
        self.model_xml_path = Path(model_xml_path).expanduser().resolve()
        if not self.model_xml_path.is_file():
            raise SimulationError(f"MuJoCo model XML does not exist: {self.model_xml_path}")

        self._mujoco = mujoco_module if mujoco_module is not None else _import_mujoco()
        self.model = self._mujoco.MjModel.from_xml_path(str(self.model_xml_path))
        self.data = self._mujoco.MjData(self.model)

    @classmethod
    def from_robot_config(
        cls,
        robot_config: Mapping[str, Any],
        *,
        repo_root: str | Path = ".",
        mujoco_module: Any | None = None,
    ) -> MujocoEnvironment:
        """Create an environment from `configs/robot/*.yaml` contents."""

        asset = _expect_mapping(robot_config, "asset")
        asset_root = asset.get("root")
        model_xml = asset.get("model_xml")

        if not isinstance(asset_root, str) or not asset_root:
            raise SimulationError("robot.asset.root must be a non-empty path string")
        if not isinstance(model_xml, str) or not model_xml:
            raise SimulationError("robot.asset.model_xml must be set before loading MuJoCo")

        model_path = Path(repo_root).expanduser().resolve() / asset_root / model_xml
        return cls(model_path, mujoco_module=mujoco_module)

    def reset(
        self,
        *,
        qpos: list[float] | tuple[float, ...] | None = None,
        qvel: list[float] | tuple[float, ...] | None = None,
        ctrl: list[float] | tuple[float, ...] | None = None,
    ) -> SimulationState:
        """Reset simulation data and optionally set initial state/control."""

        self._mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            _assign_vector(self.data.qpos, qpos, expected_len=self.model.nq, name="qpos")
        if qvel is not None:
            _assign_vector(self.data.qvel, qvel, expected_len=self.model.nv, name="qvel")
        if ctrl is not None:
            self.set_control(ctrl)
        self._mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_control(self, action: list[float] | tuple[float, ...]) -> None:
        """Assign a MuJoCo control vector after length validation."""

        _assign_vector(self.data.ctrl, action, expected_len=self.model.nu, name="ctrl")

    def step(
        self,
        action: list[float] | tuple[float, ...] | None = None,
        *,
        n_steps: int = 1,
    ) -> SimulationState:
        """Advance MuJoCo by one or more internal simulation steps."""

        if n_steps < 1:
            raise SimulationError("n_steps must be >= 1")
        if action is not None:
            self.set_control(action)

        for _ in range(n_steps):
            self._mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self) -> SimulationState:
        """Return a Python-owned snapshot of the current MuJoCo state."""

        return SimulationState(
            time_s=float(self.data.time),
            qpos=_to_float_tuple(self.data.qpos),
            qvel=_to_float_tuple(self.data.qvel),
            ctrl=_to_float_tuple(self.data.ctrl),
        )

    def summary(self) -> MujocoModelSummary:
        """Return model metadata for the SO-101 asset verification step."""

        return MujocoModelSummary(
            xml_path=self.model_xml_path,
            nq=int(self.model.nq),
            nv=int(self.model.nv),
            nu=int(self.model.nu),
            nbody=int(self.model.nbody),
            ngeom=int(self.model.ngeom),
            nsite=int(self.model.nsite),
            ncam=int(self.model.ncam),
            timestep_s=float(self.model.opt.timestep),
            joint_names=_model_names(
                self._mujoco,
                self.model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.njnt),
            ),
            site_names=_model_names(
                self._mujoco,
                self.model,
                self._mujoco.mjtObj.mjOBJ_SITE,
                int(self.model.nsite),
            ),
            camera_names=_model_names(
                self._mujoco,
                self.model,
                self._mujoco.mjtObj.mjOBJ_CAMERA,
                int(self.model.ncam),
            ),
        )


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise MujocoImportError(
            "MuJoCo is not installed. Install it on the execution machine before "
            "loading a real SO-101 model."
        ) from exc
    return mujoco


def _expect_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise SimulationError(f"robot.{key} must be a mapping")
    return value


def _assign_vector(
    target: Any,
    values: list[float] | tuple[float, ...],
    *,
    expected_len: int,
    name: str,
) -> None:
    if len(values) != expected_len:
        raise SimulationError(f"{name} must have length {expected_len}, got {len(values)}")
    for index, value in enumerate(values):
        target[index] = float(value)


def _to_float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _model_names(mujoco: Any, model: Any, object_type: Any, count: int) -> tuple[str, ...]:
    names = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        if name is not None:
            names.append(str(name))
    return tuple(names)
