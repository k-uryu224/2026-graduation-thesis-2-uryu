"""Check whether a configured MuJoCo model can be loaded."""

from __future__ import annotations

import argparse
from pathlib import Path

from so101_vla_response.configuration import ConfigError, load_experiment_bundle
from so101_vla_response.simulation import MujocoEnvironment, SimulationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/response_comparison.yaml",
        help="Path to the experiment YAML that references the robot config.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve asset paths.",
    )
    args = parser.parse_args()

    try:
        bundle = load_experiment_bundle(Path(args.experiment_config))
        env = MujocoEnvironment.from_robot_config(
            bundle.section("robot").root,
            repo_root=Path(args.repo_root),
        )
    except (ConfigError, SimulationError) as exc:
        print(f"Environment check failed: {exc}")
        return 1

    summary = env.summary()
    print(f"Loaded MuJoCo model: {summary.xml_path}")
    print(
        "Model sizes: "
        f"nq={summary.nq}, nv={summary.nv}, nu={summary.nu}, "
        f"nbody={summary.nbody}, ngeom={summary.ngeom}, "
        f"nsite={summary.nsite}, ncam={summary.ncam}"
    )
    print(f"MuJoCo timestep: {summary.timestep_s} s")
    print(f"Joints: {', '.join(summary.joint_names) or '(none)'}")
    print(f"Sites: {', '.join(summary.site_names) or '(none)'}")
    print(f"Cameras: {', '.join(summary.camera_names) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
