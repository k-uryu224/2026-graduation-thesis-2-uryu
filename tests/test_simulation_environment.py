from pathlib import Path

import pytest

from so101_vla_response.simulation import MujocoEnvironment, SimulationError


class _FakeModel:
    nq = 2
    nv = 2
    nu = 1
    nbody = 3
    ngeom = 4
    nsite = 1
    ncam = 1
    njnt = 2

    class opt:
        timestep = 0.01

    @classmethod
    def from_xml_path(cls, path: str) -> "_FakeModel":
        model = cls()
        model.loaded_path = path
        return model


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        self.time = 0.0
        self.qpos = [0.0 for _ in range(model.nq)]
        self.qvel = [0.0 for _ in range(model.nv)]
        self.ctrl = [0.0 for _ in range(model.nu)]


class _FakeMjtObj:
    mjOBJ_JOINT = "joint"
    mjOBJ_SITE = "site"
    mjOBJ_CAMERA = "camera"


class _FakeMujoco:
    MjModel = _FakeModel
    MjData = _FakeData
    mjtObj = _FakeMjtObj

    _names = {
        "joint": ("shoulder_pan", "shoulder_lift"),
        "site": ("gripperframe",),
        "camera": ("top",),
    }

    @staticmethod
    def mj_resetData(model: _FakeModel, data: _FakeData) -> None:
        data.time = 0.0
        data.qpos = [0.0 for _ in range(model.nq)]
        data.qvel = [0.0 for _ in range(model.nv)]
        data.ctrl = [0.0 for _ in range(model.nu)]

    @staticmethod
    def mj_forward(model: _FakeModel, data: _FakeData) -> None:
        del model, data

    @staticmethod
    def mj_step(model: _FakeModel, data: _FakeData) -> None:
        data.time += model.opt.timestep
        data.qpos[0] += data.ctrl[0]

    @classmethod
    def mj_id2name(cls, model: _FakeModel, object_type: str, index: int) -> str | None:
        del model
        return cls._names[object_type][index]


def test_mujoco_environment_resets_steps_and_summarizes(tmp_path: Path) -> None:
    model_path = tmp_path / "scene.xml"
    model_path.write_text("<mujoco />\n", encoding="utf-8")

    env = MujocoEnvironment(model_path, mujoco_module=_FakeMujoco)
    reset_state = env.reset(qpos=(0.1, 0.2), qvel=(0.0, 0.0), ctrl=(0.3,))
    step_state = env.step((0.5,), n_steps=2)
    summary = env.summary()

    assert reset_state.qpos == (0.1, 0.2)
    assert reset_state.ctrl == (0.3,)
    assert step_state.time_s == pytest.approx(0.02)
    assert step_state.qpos == pytest.approx((1.1, 0.2))
    assert summary.joint_names == ("shoulder_pan", "shoulder_lift")
    assert summary.site_names == ("gripperframe",)
    assert summary.camera_names == ("top",)


def test_mujoco_environment_validates_control_length(tmp_path: Path) -> None:
    model_path = tmp_path / "scene.xml"
    model_path.write_text("<mujoco />\n", encoding="utf-8")

    env = MujocoEnvironment(model_path, mujoco_module=_FakeMujoco)

    with pytest.raises(SimulationError, match="ctrl must have length 1"):
        env.set_control((0.1, 0.2))


def test_robot_config_requires_model_xml() -> None:
    with pytest.raises(SimulationError, match="model_xml"):
        MujocoEnvironment.from_robot_config(
            {
                "asset": {
                    "root": "assets/robots/so101",
                    "model_xml": None,
                }
            },
            mujoco_module=_FakeMujoco,
        )
