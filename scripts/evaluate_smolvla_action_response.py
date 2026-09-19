#!/usr/bin/env python3

import csv
import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

try:
    from lerobot.utils.control_utils import prepare_observation_for_inference
except ImportError:
    from lerobot.common.control_utils import prepare_observation_for_inference


ROOT = Path(__file__).resolve().parents[1]

GENERATOR_PATH = ROOT / "scripts/generate_lerobot_train_dataset.py"

DATASET_CONFIG_PATH = ROOT / "configs/dataset/so101_top_wrist.yaml"
TASK_CONFIG_PATH = ROOT / "configs/task/pick_object.yaml"

CHECKPOINT = (
    ROOT
    / "outputs/train/smolvla_baseline_v1"
    / "checkpoints/030000/pretrained_model"
)

OUTPUT_DIR = (
    ROOT
    / "results/evaluation/smolvla_baseline_v1"
    / "action_response_v00_seed20260920"
)

OBJECT_X_M = np.asarray(
    [0.225, 0.230, 0.235, 0.240, 0.245, 0.250, 0.255],
    dtype=np.float64,
)

BASELINE_X_M = 0.240
VARIATION_ID = "v00"
NOISE_SEED = 20260920

TASK_TEXT = "Pick up the object."
ROBOT_TYPE = "so101_mujoco"

RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "so101_train_dataset_generator",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {GENERATOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_raw_observation(
    *,
    gen,
    physics,
    model,
    data,
    renderer,
    dataset_config,
    task,
    variation,
    object_x,
):
    object_position = np.asarray(
        [
            object_x,
            gen.OBJECT_Y_M,
            gen.OBJECT_Z_M,
        ],
        dtype=float,
    )

    _, trajectory = gen.prepare_episode(
        physics=physics,
        model=model,
        data=data,
        task=task,
        object_position=object_position,
        variation=variation,
    )

    arm_actuators = [
        gen.actuator_id(model, name)
        for name in physics.ARM_JOINT_NAMES
    ]

    gripper_actuator = gen.actuator_id(
        model,
        physics.GRIPPER_JOINT_NAME,
    )

    home_arm = trajectory.arm_qpos[0]

    for actuator, value in zip(
        arm_actuators,
        home_arm,
        strict=True,
    ):
        data.ctrl[actuator] = float(value)

    data.ctrl[gripper_actuator] = gen.OPEN_CTRL

    mujoco.mj_forward(model, data)

    joint_names = list(
        dataset_config["state"]["mujoco_joints"]
    )

    state = gen.get_state(
        model,
        data,
        joint_names,
    ).copy()

    observation = {
        "observation.state": state,
    }

    rendered_images = {}

    for camera in dataset_config["cameras"].values():
        camera_name = camera["mujoco_name"]
        lerobot_key = camera["lerobot_key"]

        renderer.update_scene(
            data,
            camera=camera_name,
        )

        image = renderer.render().copy()

        observation[lerobot_key] = image
        rendered_images[lerobot_key] = image

    return observation, rendered_images


def process_observation(
    *,
    observation,
    preprocessor,
    device,
):
    # prepare_observation_for_inference may mutate the input dict.
    # Keep the raw MuJoCo observation unchanged for evaluation logging.
    batch = prepare_observation_for_inference(
        observation.copy(),
        device,
        TASK_TEXT,
        ROBOT_TYPE,
    )

    return preprocessor(batch)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert CHECKPOINT.is_dir(), CHECKPOINT

    gen = load_generator_module()
    physics = gen.load_physics_module()

    dataset_config = load_yaml(
        DATASET_CONFIG_PATH
    )

    task_doc = load_yaml(
        TASK_CONFIG_PATH
    )
    task = task_doc["task"]

    variations = gen.load_variations()

    variation = next(
        item
        for item in variations
        if item["id"] == VARIATION_ID
    )

    print("=== evaluation condition ===")
    print("checkpoint:", CHECKPOINT)
    print("variation:", VARIATION_ID)
    print("noise seed:", NOISE_SEED)
    print("baseline x:", BASELINE_X_M)
    print("positions:", OBJECT_X_M.tolist())

    device = torch.device("cuda")

    print("\n=== load policy ===")

    policy = SmolVLAPolicy.from_pretrained(
        str(CHECKPOINT)
    )
    policy = policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(CHECKPOINT),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
            "rename_observations_processor": {
                "rename_map": RENAME_MAP,
            },
        },
    )

    noise_shape = (
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
    )

    noise_generator = torch.Generator(
        device=device
    )
    noise_generator.manual_seed(
        NOISE_SEED
    )

    fixed_noise = torch.randn(
        noise_shape,
        generator=noise_generator,
        device=device,
        dtype=torch.float32,
    )

    print("fixed noise shape:", tuple(fixed_noise.shape))

    print("\n=== load MuJoCo ===")

    model = mujoco.MjModel.from_xml_path(
        str(gen.SCENE_PATH)
    )
    data = mujoco.MjData(model)

    height = int(
        dataset_config["images"]["height"]
    )
    width = int(
        dataset_config["images"]["width"]
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    states = []
    raw_chunks = []
    action_chunks = []
    top_images = []
    wrist_images = []

    try:
        for object_x in OBJECT_X_M:
            print(
                f"\n--- x={object_x:.3f} m ---"
            )

            observation, images = build_raw_observation(
                gen=gen,
                physics=physics,
                model=model,
                data=data,
                renderer=renderer,
                dataset_config=dataset_config,
                task=task,
                variation=variation,
                object_x=float(object_x),
            )

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            batch = process_observation(
                observation=observation,
                preprocessor=preprocessor,
                device=device,
            )

            with torch.inference_mode():
                raw_chunk = policy.predict_action_chunk(
                    batch,
                    noise=fixed_noise.clone(),
                )

                action_chunk = postprocessor(
                    raw_chunk.clone()
                )

            raw_np = (
                raw_chunk[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            action_np = (
                action_chunk[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            state_np = (
                observation["observation.state"]
                .copy()
                .astype(np.float32)
            )

            states.append(state_np)
            raw_chunks.append(raw_np)
            action_chunks.append(action_np)

            top_images.append(
                images[
                    "observation.images.top"
                ]
            )

            wrist_images.append(
                images[
                    "observation.images.wrist"
                ]
            )

            print("state:", state_np)
            print("first action:", action_np[0])

    finally:
        renderer.close()

    states = np.stack(states)
    raw_chunks = np.stack(raw_chunks)
    action_chunks = np.stack(action_chunks)

    top_images = np.stack(top_images)
    wrist_images = np.stack(wrist_images)

    baseline_candidates = np.where(
        np.isclose(
            OBJECT_X_M,
            BASELINE_X_M,
            atol=1e-9,
        )
    )[0]

    if len(baseline_candidates) != 1:
        raise RuntimeError(
            "Could not identify baseline position."
        )

    baseline_index = int(
        baseline_candidates[0]
    )

    raw_responses = (
        raw_chunks
        - raw_chunks[baseline_index]
    )

    responses = (
        action_chunks
        - action_chunks[baseline_index]
    )

    state_diff = np.max(
        np.abs(
            states
            - states[baseline_index]
        ),
        axis=1,
    )

    if np.max(state_diff) > 1e-6:
        raise RuntimeError(
            "Robot state changed across object positions: "
            f"{state_diff}"
        )

    npz_path = (
        OUTPUT_DIR
        / "action_response.npz"
    )

    np.savez_compressed(
        npz_path,
        object_x_m=OBJECT_X_M,
        delta_x_m=OBJECT_X_M - BASELINE_X_M,
        states=states,
        raw_action_chunks=raw_chunks,
        action_chunks=action_chunks,
        raw_responses=raw_responses,
        responses=responses,
        top_images=top_images,
        wrist_images=wrist_images,
        fixed_noise=(
            fixed_noise
            .detach()
            .cpu()
            .numpy()
        ),
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    summary_rows = []

    for i, object_x in enumerate(OBJECT_X_M):
        response = responses[i]

        row = {
            "object_x_m": float(object_x),
            "delta_x_cm": float(
                (object_x - BASELINE_X_M) * 100.0
            ),
            "state_max_abs_diff": float(
                state_diff[i]
            ),
            "response_rms": float(
                np.sqrt(
                    np.mean(
                        response ** 2
                    )
                )
            ),
            "response_mean_abs": float(
                np.mean(
                    np.abs(response)
                )
            ),
            "response_first_l2": float(
                np.linalg.norm(
                    response[0]
                )
            ),
        }

        for j, name in enumerate(ACTION_NAMES):
            row[f"action0_{name}"] = float(
                action_chunks[i, 0, j]
            )
            row[f"response0_{name}"] = float(
                response[0, j]
            )

        summary_rows.append(row)

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    long_path = (
        OUTPUT_DIR
        / "action_response_long.csv"
    )

    with long_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "object_x_m",
            "delta_x_cm",
            "step",
            "action_index",
            "action_name",
            "action",
            "baseline_action",
            "response",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        baseline_chunk = (
            action_chunks[baseline_index]
        )

        for pos_idx, object_x in enumerate(
            OBJECT_X_M
        ):
            for step in range(
                policy.config.chunk_size
            ):
                for action_index, action_name in enumerate(
                    ACTION_NAMES
                ):
                    writer.writerow(
                        {
                            "object_x_m": float(object_x),
                            "delta_x_cm": float(
                                (
                                    object_x
                                    - BASELINE_X_M
                                )
                                * 100.0
                            ),
                            "step": step,
                            "action_index": action_index,
                            "action_name": action_name,
                            "action": float(
                                action_chunks[
                                    pos_idx,
                                    step,
                                    action_index,
                                ]
                            ),
                            "baseline_action": float(
                                baseline_chunk[
                                    step,
                                    action_index,
                                ]
                            ),
                            "response": float(
                                responses[
                                    pos_idx,
                                    step,
                                    action_index,
                                ]
                            ),
                        }
                    )

    metadata = {
        "checkpoint": str(CHECKPOINT),
        "task": TASK_TEXT,
        "robot_type": ROBOT_TYPE,
        "variation_id": VARIATION_ID,
        "noise_seed": NOISE_SEED,
        "baseline_x_m": BASELINE_X_M,
        "object_x_m": OBJECT_X_M.tolist(),
        "chunk_size": int(
            policy.config.chunk_size
        ),
        "action_dim": len(ACTION_NAMES),
        "action_names": ACTION_NAMES,
        "response_definition": (
            "A(x) - A(0.240), using identical "
            "initial robot state and identical inference noise"
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Action Response summary ===")

    for row in summary_rows:
        print(
            f"x={row['object_x_m']:.3f} "
            f"delta={row['delta_x_cm']:+.1f} cm "
            f"RMS={row['response_rms']:.6f} "
            f"first_L2={row['response_first_l2']:.6f}"
        )

    print("\nstate max abs diff:", float(np.max(state_diff)))

    print("\nSaved:")
    print(npz_path)
    print(summary_path)
    print(long_path)
    print(metadata_path)

    print()
    print("SMOLVLA ACTION RESPONSE EVALUATION PASSED")


if __name__ == "__main__":
    main()
