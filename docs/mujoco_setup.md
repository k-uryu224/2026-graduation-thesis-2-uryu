# MuJoCo Setup

## Purpose

Before loading an SO-101 robot asset, first confirm that MuJoCo itself works on
the execution machine.

This step checks only the simulator installation and Python bindings:

- Python can import `mujoco`
- a minimal XML model can be loaded
- `reset` and `step` run without a robot asset
- the repository's thin environment wrapper works with a real MuJoCo backend

## Install on cicero

From the repository root:

```bash
cd ~/2026-graduation-thesis-2-uryu
uv sync --extra dev
uv pip install mujoco
```

The `mujoco` Python package provides the official Python bindings and includes
the MuJoCo library, so a separate MuJoCo download is not required for normal
Python usage.

## Smoke test

Run:

```bash
uv run python scripts/smoke_mujoco.py
```

Expected output should include:

```text
MuJoCo smoke test passed.
```

This smoke test uses a temporary XML file with a falling box and ground plane.
It does not use SO-101, cameras, IK, LeRobot, or any model checkpoint.

## Current SO-101 status

`scripts/check_environment.py` is for the real SO-101 model. It will fail until
`configs/robot/so101.yaml` has a valid `robot.asset.model_xml`.

That failure is expected at this stage:

```text
Environment check failed: robot.asset.model_xml must be set before loading MuJoCo
```

After the SO-101 XML/MJCF and referenced mesh files are placed under
`assets/robots/so101/`, update:

```yaml
robot:
  asset:
    root: assets/robots/so101
    model_xml: <actual XML file>
```

Then run:

```bash
uv run python scripts/check_environment.py
```

The next research step is to record the actual joint, site, and camera names
reported by that command.
