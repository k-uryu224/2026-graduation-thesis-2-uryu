"""Run the fixed 20 x 3 variation regression through canonical grasp code."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from so101_vla_response.configuration import (
    load_experiment_bundle,
)
from so101_vla_response.simulation import (
    MujocoEnvironment,
)

from check_grasp_physics import (
    AXIS_TO_INDEX,
    GRIPPER_JOINT_NAME,
    GRIPPER_LINK_BODY_NAME,
    JAW_LINK_BODY_NAME,
    TARGET_BOX_BODY_NAME,
    actuator_id_for_joint,
    arm_actuator_ids,
    descendant_body_ids,
    load_yaml,
    object_geom_ids,
    run_one,
)


FIXTURE_PATH = Path(
    "configs/trajectory/"
    "grasp_variations_v1.json"
)

OUTPUT_DIR = Path(
    "results/calibration/"
    "official_variation_regression_v1"
)


def fixture_entries(
    payload,
):
    """Return (variation_id, dict) entries from common fixture layouts."""

    if isinstance(
        payload,
        list,
    ):
        return [
            (
                item.get(
                    "id",
                    item.get(
                        "variation_id",
                        f"v{index:02d}",
                    ),
                ),
                item,
            )
            for index, item
            in enumerate(
                payload
            )
        ]

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Variation fixture must be "
            "a list or mapping."
        )

    for key in (
        "variations",
        "samples",
        "items",
        "entries",
    ):
        value = payload.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return [
                (
                    item.get(
                        "id",
                        item.get(
                            "variation_id",
                            f"v{index:02d}",
                        ),
                    ),
                    item,
                )
                for index, item
                in enumerate(
                    value
                )
            ]

        if isinstance(
            value,
            dict,
        ):
            return [
                (
                    str(
                        variation_id
                    ),
                    item,
                )
                for (
                    variation_id,
                    item,
                ) in value.items()
                if isinstance(
                    item,
                    dict,
                )
            ]

    direct = [
        (
            str(
                variation_id
            ),
            item,
        )
        for (
            variation_id,
            item,
        ) in payload.items()
        if (
            isinstance(
                item,
                dict,
            )
            and str(
                variation_id
            ).startswith(
                "v"
            )
        )
    ]

    if direct:
        return direct

    raise ValueError(
        "Could not locate variation entries "
        f"in fixture keys: {list(payload)}"
    )


def get_home_offsets_deg(
    item: dict,
) -> np.ndarray:
    for key in (
        "home_joint_offsets_deg",
        "home_offsets_deg",
    ):
        if key in item:
            values = item[
                key
            ]

            result = np.asarray(
                values,
                dtype=float,
            )

            if result.shape != (
                5,
            ):
                raise ValueError(
                    f"{key} must have shape (5,), "
                    f"got {result.shape}."
                )

            return result

    home = item.get(
        "home"
    )

    if isinstance(
        home,
        dict,
    ):
        for key in (
            "joint_offsets_deg",
            "offsets_deg",
        ):
            if key in home:
                result = np.asarray(
                    home[
                        key
                    ],
                    dtype=float,
                )

                if result.shape != (
                    5,
                ):
                    raise ValueError(
                        "Home offsets must "
                        "have shape (5,)."
                    )

                return result

    raise ValueError(
        "Home joint offsets were not found. "
        f"Available keys: {list(item)}"
    )


def get_pre_z_offset_m(
    item: dict,
) -> float:
    for key in (
        "pregrasp_z_offset_m",
        "pre_grasp_z_offset_m",
        "pre_z_offset_m",
    ):
        if key in item:
            return float(
                item[
                    key
                ]
            )

    for key in (
        "pregrasp_z_offset_mm",
        "pre_grasp_z_offset_mm",
        "pre_z_offset_mm",
    ):
        if key in item:
            return (
                float(
                    item[
                        key
                    ]
                )
                / 1000.0
            )

    pre = item.get(
        "pre_grasp"
    )

    if isinstance(
        pre,
        dict,
    ):
        if "z_offset_m" in pre:
            return float(
                pre[
                    "z_offset_m"
                ]
            )

        if "z_offset_mm" in pre:
            return (
                float(
                    pre[
                        "z_offset_mm"
                    ]
                )
                / 1000.0
            )

    raise ValueError(
        "Pre-grasp z variation was not found. "
        f"Available keys: {list(item)}"
    )


def main() -> int:
    fixture_payload = json.loads(
        FIXTURE_PATH.read_text(
            encoding="utf-8"
        )
    )

    variations = fixture_entries(
        fixture_payload
    )

    task = load_yaml(
        Path(
            "configs/task/"
            "pick_object.yaml"
        )
    )[
        "task"
    ]

    expected_variations = int(
        task[
            "variation"
        ][
            "count"
        ]
    )

    if len(
        variations
    ) != expected_variations:
        raise RuntimeError(
            "Unexpected variation count: "
            f"{len(variations)} "
            f"(expected "
            f"{expected_variations})"
        )

    bundle = load_experiment_bundle(
        Path(
            "configs/experiment/"
            "response_comparison.yaml"
        )
    )

    env = (
        MujocoEnvironment
        .from_robot_config(
            bundle.section(
                "robot"
            ).root,
            repo_root=Path("."),
        )
    )

    env.reset()

    model = env.model
    data = env.data

    scene = load_yaml(
        Path(
            "configs/scene/"
            "top_grasp.yaml"
        )
    )[
        "scene"
    ]

    perturbation = scene[
        "perturbation"
    ]

    perturb_index = (
        AXIS_TO_INDEX[
            perturbation[
                "axis"
            ]
        ]
    )

    training_offsets = (
        perturbation[
            "training_offsets_m"
        ]
    )

    baseline_position = np.asarray(
        scene[
            "object"
        ][
            "baseline_pose"
        ][
            "position_m"
        ],
        dtype=float,
    )

    object_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "target_box_center",
    )

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
        object_site_id,
        target_box_body_id,
        gripper_link_body_id,
        jaw_link_body_id,
    ) < 0:
        raise RuntimeError(
            "Required MuJoCo object not found."
        )

    object_bodies = (
        descendant_body_ids(
            model,
            target_box_body_id,
        )
    )

    jaw_bodies = (
        descendant_body_ids(
            model,
            jaw_link_body_id,
        )
    )

    gripper_bodies = (
        descendant_body_ids(
            model,
            gripper_link_body_id,
        )
    )

    fixed_gripper_bodies = (
        gripper_bodies
        - jaw_bodies
    )

    object_geoms = (
        object_geom_ids(
            model,
            object_bodies,
        )
    )

    moving_geoms = (
        object_geom_ids(
            model,
            jaw_bodies,
        )
    )

    fixed_geoms = (
        object_geom_ids(
            model,
            fixed_gripper_bodies,
        )
    )

    arm_actuators = (
        arm_actuator_ids(
            model
        )
    )

    gripper_joint_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            GRIPPER_JOINT_NAME,
        )
    )

    gripper_actuator = (
        actuator_id_for_joint(
            model,
            gripper_joint_id,
        )
    )

    # Confirmed mapping.
    mapping_name = (
        "candidate_A_upper_open"
    )

    open_ctrl = 1.0
    closed_ctrl = 0.0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    passed = 0
    total = 0

    print(
        "SO-101 official fixed-variation "
        "physics regression"
    )

    print()
    print(
        "Canonical flow:"
    )
    print(
        "  pick_object.yaml"
    )
    print(
        "  -> build_waypoint_trajectory()"
    )
    print(
        "  -> Reverse-Lift approach"
    )
    print(
        "  -> scripted_grasp.py"
    )
    print(
        "  -> run_one() MuJoCo physics"
    )
    print()

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

        object_x_mm = (
            object_position[
                0
            ]
            * 1000.0
        )

        for (
            variation_id,
            variation,
        ) in variations:
            home_deg = (
                get_home_offsets_deg(
                    variation
                )
            )

            pre_z_m = (
                get_pre_z_offset_m(
                    variation
                )
            )

            result = run_one(
                model=model,
                data=data,
                task=task,
                object_position=(
                    object_position
                ),
                object_offset=float(
                    object_offset
                ),
                mapping_name=(
                    mapping_name
                ),
                open_ctrl=open_ctrl,
                closed_ctrl=(
                    closed_ctrl
                ),
                arm_actuators=(
                    arm_actuators
                ),
                gripper_actuator=(
                    gripper_actuator
                ),
                object_site_id=(
                    object_site_id
                ),
                object_geoms=(
                    object_geoms
                ),
                moving_geoms=(
                    moving_geoms
                ),
                fixed_geoms=(
                    fixed_geoms
                ),
                home_joint_offsets_rad=(
                    np.deg2rad(
                        home_deg
                    )
                ),
                pre_grasp_z_offset_m=(
                    pre_z_m
                ),
            )

            trial_pass = bool(
                result.success
                and result
                .moving_jaw_contact_seen
                and result
                .fixed_finger_contact_seen
                and result
                .simultaneous_two_side_contact_seen
                and not result
                .pre_close_contact_seen
            )

            total += 1

            if trial_pass:
                passed += 1

            status = (
                "PASS"
                if trial_pass
                else "FAIL"
            )

            print(
                f"x={object_x_mm:.0f} "
                f"{variation_id} "
                f"pre_z="
                f"{pre_z_m * 1000:+.2f} mm "
                f"final="
                f"{result.final_lift_m * 1000:.2f} "
                f"two_side="
                f"{int(result.simultaneous_two_side_contact_seen)} "
                f"preclose="
                f"{int(result.pre_close_contact_seen)} "
                f"{status}"
            )

            rows.append(
                {
                    "object_x_m": (
                        object_position[
                            0
                        ]
                    ),
                    "variation_id": (
                        variation_id
                    ),
                    "home_joint_offsets_deg": (
                        json.dumps(
                            home_deg.tolist()
                        )
                    ),
                    "pre_grasp_z_offset_m": (
                        pre_z_m
                    ),
                    "success": (
                        result.success
                    ),
                    "moving_contact": (
                        result
                        .moving_jaw_contact_seen
                    ),
                    "fixed_contact": (
                        result
                        .fixed_finger_contact_seen
                    ),
                    "two_side_contact": (
                        result
                        .simultaneous_two_side_contact_seen
                    ),
                    "pre_close_contact": (
                        result
                        .pre_close_contact_seen
                    ),
                    "final_lift_m": (
                        result.final_lift_m
                    ),
                    "max_lift_m": (
                        result.max_lift_m
                    ),
                    "hold_frames": (
                        result.hold_frames
                    ),
                    "required_hold_frames": (
                        result.required_hold_frames
                    ),
                    "pass": trial_pass,
                }
            )

    csv_path = (
        OUTPUT_DIR
        / "results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    expected_total = (
        len(
            training_offsets
        )
        * expected_variations
    )

    print()
    print("=" * 80)
    print(
        f"PASS: {passed}/{total}"
    )
    print(
        f"FAIL: {total - passed}/{total}"
    )
    print(
        f"CSV: {csv_path}"
    )

    if total != expected_total:
        print(
            "RESULT: FAIL - unexpected "
            "trial count."
        )

        return 1

    if passed != expected_total:
        print(
            "RESULT: FAIL - canonical "
            "variation regression failed."
        )

        return 1

    print()
    print(
        "RESULT: PASS - all fixed variations "
        "succeeded through canonical code, "
        "with two-side grasp and no "
        "pre-close object contact."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
