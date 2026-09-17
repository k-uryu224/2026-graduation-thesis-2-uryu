"""Inspect important coordinates in the configured MuJoCo scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from so101_vla_response.configuration import ConfigError, load_experiment_bundle
from so101_vla_response.simulation import MujocoEnvironment, SimulationError


SITE_NAMES = (
    "baseframe",
    "gripperframe",
    "table_top_center",
    "target_box_center",
)


def _format_xyz(values) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/response_comparison.yaml",
        help="Path to the experiment YAML.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve asset paths.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Print coordinates without opening the MuJoCo Viewer.",
    )
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        print(f"Scene check failed: MuJoCo is not installed: {exc}")
        return 1

    try:
        bundle = load_experiment_bundle(Path(args.experiment_config))
        env = MujocoEnvironment.from_robot_config(
            bundle.section("robot").root,
            repo_root=Path(args.repo_root),
        )
        env.reset()
    except (ConfigError, SimulationError) as exc:
        print(f"Scene check failed: {exc}")
        return 1

    print(f"Loaded scene: {env.model_xml_path}")

    print()
    print("World origin")
    print("  world                [0.000000, 0.000000, 0.000000]")

    print()
    print("Important sites")

    positions: dict[str, tuple[float, float, float]] = {}

    for name in SITE_NAMES:
        site_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id < 0:
            print(f"  {name:<20} NOT FOUND")
            continue

        position = tuple(float(value) for value in env.data.site_xpos[site_id])
        positions[name] = position

        print(f"  {name:<20} {_format_xyz(position)}")

    table_position = positions.get("table_top_center")
    object_position = positions.get("target_box_center")

    if table_position is not None and object_position is not None:
        delta = tuple(
            object_value - table_value
            for object_value, table_value in zip(
                object_position,
                table_position,
                strict=True,
            )
        )

        print()
        print("Object relative to table top")
        print(f"  target - table       {_format_xyz(delta)}")

    gripper_site_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripperframe",
    )

    if gripper_site_id >= 0:
        rotation = env.data.site_xmat[gripper_site_id].reshape(3, 3)

        print()
        print("Gripperframe orientation")
        print("  rotation matrix (local -> world)")

        for row in rotation:
            print(
                "  ["
                + ", ".join(f"{float(value): .6f}" for value in row)
                + "]"
            )

        print()
        print("Gripperframe local axes in world coordinates")
        print(f"  local X = {_format_xyz(rotation[:, 0])}")
        print(f"  local Y = {_format_xyz(rotation[:, 1])}")
        print(f"  local Z = {_format_xyz(rotation[:, 2])}")

    if args.no_viewer:
        return 0

    try:
        import mujoco.viewer

        print()
        print("Opening MuJoCo Viewer...")
        print("Close the viewer window to finish.")
        mujoco.viewer.launch(env.model, env.data)
    except Exception as exc:
        print(f"Viewer launch failed: {exc}")
        print("Coordinate inspection completed successfully.")
        print("Use --no-viewer on a headless machine.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
