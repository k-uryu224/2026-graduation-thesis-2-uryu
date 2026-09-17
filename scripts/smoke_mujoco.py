"""Run a minimal MuJoCo smoke test without any SO-101 assets."""

from __future__ import annotations

import tempfile
from pathlib import Path

from so101_vla_response.simulation import MujocoEnvironment, SimulationError

SMOKE_MODEL_XML = """\
<mujoco model="smoke_test">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.2">
      <freejoint name="box_freejoint"/>
      <geom name="box_geom" type="box" size="0.03 0.03 0.03" mass="0.1" rgba="0.2 0.4 0.8 1"/>
    </body>
    <site name="box_center" pos="0 0 0.2" size="0.01" rgba="1 0 0 1"/>
    <camera name="overview" pos="0 -0.7 0.35" xyaxes="1 0 0 0 0.4 1"/>
  </worldbody>
</mujoco>
"""


def main() -> int:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "smoke_model.xml"
            xml_path.write_text(SMOKE_MODEL_XML, encoding="utf-8")

            env = MujocoEnvironment(xml_path)
            reset_state = env.reset()
            step_state = env.step(n_steps=10)
            summary = env.summary()
    except SimulationError as exc:
        print(f"MuJoCo smoke test failed: {exc}")
        return 1

    print("MuJoCo smoke test passed.")
    print(f"Model: {summary.xml_path}")
    print(f"nq={summary.nq}, nv={summary.nv}, nu={summary.nu}, timestep={summary.timestep_s}")
    print(f"joints={summary.joint_names}")
    print(f"sites={summary.site_names}")
    print(f"cameras={summary.camera_names}")
    print(f"time: {reset_state.time_s:.3f} -> {step_state.time_s:.3f} s")
    print(f"qpos length: {len(step_state.qpos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
