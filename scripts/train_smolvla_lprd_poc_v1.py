#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

try:
    from lerobot.utils.control_utils import (
        prepare_observation_for_inference,
    )
except ImportError:
    from lerobot.common.control_utils import (
        prepare_observation_for_inference,
    )


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROTOCOL = (
    ROOT
    / "configs/training/"
    "lprd_shoulder_lift_poc_v1_finetune_protocol.json"
)

TARGET_MODES = (
    "no_response",
    "pointwise_kd",
    "response_target",
)

TASK_TEXT = "Pick up the object."
ROBOT_TYPE = "so101_mujoco"

REQUIRED_BATCH_KEYS = (
    "observation.state",
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.language.tokens",
    "observation.language.attention_mask",
    "action",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )

    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(
        seed % (2**32)
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def make_raw_observation(
    data,
    index: int,
):
    return {
        "observation.state":
            data[
                "observation_state"
            ][
                index
            ]
            .copy()
            .astype(np.float32),

        "observation.images.top":
            data[
                "observation_images_top"
            ][
                index
            ]
            .copy(),

        "observation.images.wrist":
            data[
                "observation_images_wrist"
            ][
                index
            ]
            .copy(),
    }


def process_training_sample(
    *,
    data,
    index,
    target_key,
    preprocessor,
    device,
):
    observation = (
        make_raw_observation(
            data,
            index,
        )
    )

    raw = (
        prepare_observation_for_inference(
            observation,
            device,
            TASK_TEXT,
            ROBOT_TYPE,
        )
    )

    target = (
        data[target_key][index]
        .copy()
        .astype(np.float32)
    )

    if target.shape != (
        50,
        6,
    ):
        raise RuntimeError(
            f"Unexpected target shape: "
            f"{target.shape}"
        )

    # Observation already has batch dimension after
    # prepare_observation_for_inference().
    # Action is inserted afterwards and must be
    # batched manually.
    raw["action"] = (
        torch.from_numpy(
            target
        )
        .unsqueeze(0)
    )

    processed = (
        preprocessor(
            raw
        )
    )

    missing = [
        key
        for key in REQUIRED_BATCH_KEYS
        if key not in processed
    ]

    if missing:
        raise RuntimeError(
            f"Missing processed keys: "
            f"{missing}"
        )

    if processed[
        "action"
    ].shape != (
        1,
        50,
        6,
    ):
        raise RuntimeError(
            "Processed action shape "
            f"is "
            f"{processed['action'].shape}"
        )

    return {
        key: processed[key]
        for key in REQUIRED_BATCH_KEYS
    }


def concatenate_samples(
    samples,
):
    batch = {}

    for key in REQUIRED_BATCH_KEYS:
        values = [
            item[key]
            for item in samples
        ]

        if not all(
            torch.is_tensor(value)
            for value in values
        ):
            raise RuntimeError(
                f"{key} is not tensor-only"
            )

        batch[key] = torch.cat(
            values,
            dim=0,
        )

    return batch


def build_training_batch(
    *,
    data,
    target_key,
    preprocessor,
    device,
):
    n = int(
        data[
            "observation_state"
        ].shape[0]
    )

    samples = [
        process_training_sample(
            data=data,
            index=index,
            target_key=target_key,
            preprocessor=preprocessor,
            device=device,
        )
        for index in range(n)
    ]

    batch = concatenate_samples(
        samples
    )

    expected = {
        "observation.state":
            (n, 6),
        "action":
            (n, 50, 6),
        "observation.language.tokens":
            (n, 48),
        "observation.language.attention_mask":
            (n, 48),
    }

    for key, shape in expected.items():
        if tuple(
            batch[key].shape
        ) != shape:
            raise RuntimeError(
                f"{key}: expected "
                f"{shape}, got "
                f"{tuple(batch[key].shape)}"
            )

    return batch


def verify_identical_observations(
    data,
):
    n = int(
        data[
            "observation_state"
        ].shape[0]
    )

    if n != 10:
        raise RuntimeError(
            f"Expected 10 samples, got {n}"
        )

    offsets = (
        data[
            "shoulder_lift_offsets_deg"
        ]
        .astype(np.float64)
    )

    if np.any(
        np.isclose(
            offsets,
            -5.0,
        )
    ):
        raise RuntimeError(
            "Held-out -5deg leaked "
            "into training data."
        )

    if sorted(
        set(
            offsets.tolist()
        )
    ) != [
        -6.0,
        -4.0,
    ]:
        raise RuntimeError(
            f"Unexpected offsets: "
            f"{sorted(set(offsets.tolist()))}"
        )


def count_parameters(policy):
    total = sum(
        p.numel()
        for p in policy.parameters()
    )

    trainable = sum(
        p.numel()
        for p in policy.parameters()
        if p.requires_grad
    )

    return total, trainable


def save_checkpoint_bundle(
    *,
    save_dir,
    policy,
    preprocessor,
    postprocessor,
):
    save_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    policy.save_pretrained(
        save_dir
    )

    preprocessor.save_pretrained(
        save_dir
    )

    postprocessor.save_pretrained(
        save_dir
    )


def reload_and_inference_smoke(
    *,
    save_dir,
    data,
    device,
    seed,
):
    reloaded = (
        SmolVLAPolicy
        .from_pretrained(
            str(save_dir)
        )
        .to(device)
    )

    reloaded.eval()

    (
        reloaded_preprocessor,
        reloaded_postprocessor,
    ) = make_pre_post_processors(
        policy_cfg=reloaded.config,
        pretrained_path=str(
            save_dir
        ),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
        },
    )

    observation = (
        make_raw_observation(
            data,
            0,
        )
    )

    raw = (
        prepare_observation_for_inference(
            observation,
            device,
            TASK_TEXT,
            ROBOT_TYPE,
        )
    )

    processed = (
        reloaded_preprocessor(
            raw
        )
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        int(seed)
    )

    noise = torch.randn(
        (
            1,
            int(
                reloaded.config.chunk_size
            ),
            int(
                reloaded.config.max_action_dim
            ),
        ),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        normalized = (
            reloaded
            .predict_action_chunk(
                processed,
                noise=noise,
            )
        )

        physical = (
            reloaded_postprocessor(
                normalized.clone()
            )
        )

    if tuple(
        physical.shape
    ) != (
        1,
        50,
        6,
    ):
        raise RuntimeError(
            "Reload inference shape "
            f"unexpected: "
            f"{tuple(physical.shape)}"
        )

    if not torch.isfinite(
        physical
    ).all():
        raise RuntimeError(
            "Reloaded inference "
            "contains non-finite values."
        )

    return (
        physical[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main():
    args = parse_args()

    protocol_path = (
        args.protocol.resolve()
    )

    protocol = load_json(
        protocol_path
    )

    mode = args.target_mode

    target_modes = protocol[
        "target_modes"
    ]

    if mode not in target_modes:
        raise RuntimeError(
            f"Target mode not in protocol: "
            f"{mode}"
        )

    target_key = (
        target_modes[
            mode
        ]
    )

    data_path = (
        ROOT
        / protocol[
            "training_data"
        ]
    ).resolve()

    data_lock = (
        ROOT
        / protocol[
            "training_data_lock"
        ]
    ).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            data_path
        )

    if not data_lock.is_file():
        raise FileNotFoundError(
            data_lock
        )

    subprocess.run(
        [
            "sha256sum",
            "-c",
            str(data_lock),
        ],
        cwd=ROOT,
        check=True,
    )

    data = np.load(
        data_path,
        allow_pickle=False,
    )

    verify_identical_observations(
        data
    )

    if target_key not in data.files:
        raise RuntimeError(
            f"{target_key} not found "
            "in training data."
        )

    base_checkpoint = (
        ROOT
        / protocol[
            "base_checkpoint"
        ]
    ).resolve()

    if not base_checkpoint.is_dir():
        raise FileNotFoundError(
            base_checkpoint
        )

    optimization = (
        protocol[
            "optimization"
        ]
    )

    configured_steps = int(
        optimization[
            "steps"
        ]
    )

    steps = (
        5
        if args.smoke
        else configured_steps
    )

    if (
        not args.smoke
        and steps != 500
    ):
        raise RuntimeError(
            "Production PoC must use "
            "exactly 500 steps."
        )

    batch_size = int(
        protocol[
            "data"
        ][
            "batch_size"
        ]
    )

    if batch_size != 10:
        raise RuntimeError(
            "Expected full batch size 10."
        )

    learning_rate = float(
        optimization[
            "learning_rate"
        ]
    )

    if not math.isclose(
        learning_rate,
        1e-5,
        abs_tol=0.0,
        rel_tol=0.0,
    ):
        raise RuntimeError(
            "Protocol learning rate "
            "must be 1e-5."
        )

    training_seed = int(
        protocol[
            "stochastic_control"
        ][
            "training_seed"
        ]
    )

    output_root = (
        ROOT
        / (
            "outputs/smoke/"
            "lprd_shoulder_lift_poc_v1"
            if args.smoke
            else
            "outputs/train/"
            "lprd_shoulder_lift_poc_v1"
        )
        / mode
    )

    final_dir = (
        output_root
        / "checkpoints"
        / f"{steps:06d}"
        / "pretrained_model"
    )

    log_path = (
        output_root
        / "training_log.csv"
    )

    metadata_path = (
        output_root
        / "finetune_metadata.json"
    )

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(
                output_root
            )
        else:
            raise SystemExit(
                f"Output exists:\n"
                f"{output_root}\n"
                "Use --overwrite only "
                "if intentional."
            )

    print(
        "=== LPRD SmolVLA finetune ==="
    )

    print(
        "mode:",
        mode,
    )

    print(
        "target key:",
        target_key,
    )

    print(
        "base:",
        base_checkpoint,
    )

    print(
        "training data:",
        data_path,
    )

    print(
        "steps:",
        steps,
    )

    print(
        "batch size:",
        batch_size,
    )

    print(
        "learning rate:",
        learning_rate,
    )

    print(
        "training seed:",
        training_seed,
    )

    print(
        "smoke:",
        args.smoke,
    )

    if args.validate_only:
        print(
            "FINETUNE VALIDATION ONLY PASSED"
        )
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
    )

    # Ensure all three modes start from the
    # identical model initialization.
    set_all_seeds(
        training_seed
    )

    policy = (
        SmolVLAPolicy
        .from_pretrained(
            str(
                base_checkpoint
            )
        )
        .to(device)
    )

    if not (
        bool(
            policy.config.freeze_vision_encoder
        )
        and bool(
            policy.config.train_expert_only
        )
        and bool(
            policy.config.train_state_proj
        )
    ):
        raise RuntimeError(
            "Unexpected SmolVLA "
            "trainable-scope config."
        )

    (
        preprocessor,
        postprocessor,
    ) = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(
            base_checkpoint
        ),
        preprocessor_overrides={
            "device_processor": {
                "device": "cuda",
            },
        },
    )

    batch = build_training_batch(
        data=data,
        target_key=target_key,
        preprocessor=preprocessor,
        device=device,
    )

    if int(
        batch[
            "action"
        ].shape[0]
    ) != batch_size:
        raise RuntimeError(
            "Processed batch size mismatch."
        )

    total_params, trainable_params = (
        count_parameters(
            policy
        )
    )

    print()
    print(
        "parameters total:",
        total_params,
    )

    print(
        "parameters trainable:",
        trainable_params,
    )

    if trainable_params <= 0:
        raise RuntimeError(
            "No trainable parameters."
        )

    parameters = [
        p
        for p in policy.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=tuple(
            float(x)
            for x in optimization[
                "betas"
            ]
        ),
        eps=float(
            optimization[
                "eps"
            ]
        ),
        weight_decay=float(
            optimization[
                "weight_decay"
            ]
        ),
    )

    grad_clip_norm = float(
        optimization[
            "grad_clip_norm"
        ]
    )

    log_rows = []

    policy.train()

    for zero_step in range(
        steps
    ):
        step = zero_step + 1

        step_seed = (
            training_seed
            + zero_step
        )

        # This controls flow noise, sampled timestep,
        # and any later stochastic operation.
        set_all_seeds(
            step_seed
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        noise = (
            policy.model.sample_noise(
                (
                    batch_size,
                    int(
                        policy.config.chunk_size
                    ),
                    int(
                        policy.config.max_action_dim
                    ),
                ),
                device,
            )
        )

        time = (
            policy.model.sample_time(
                batch_size,
                device,
            )
        )

        loss, loss_dict = (
            policy.forward(
                batch,
                noise=noise,
                time=time,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss "
                f"at step {step}"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                parameters,
                grad_clip_norm,
                error_if_nonfinite=True,
            )
        )

        optimizer.step()

        row = {
            "step": step,
            "step_seed":
                step_seed,
            "loss":
                float(
                    loss.detach()
                    .cpu()
                ),
            "grad_norm":
                float(
                    grad_norm.detach()
                    .cpu()
                    if torch.is_tensor(
                        grad_norm
                    )
                    else grad_norm
                ),
            "learning_rate":
                float(
                    optimizer
                    .param_groups[0]["lr"]
                ),
        }

        log_rows.append(
            row
        )

        if (
            step == 1
            or step == steps
            or step % 25 == 0
        ):
            print(
                f"step "
                f"{step:04d}/{steps:04d} "
                f"loss="
                f"{row['loss']:.6f} "
                f"grad_norm="
                f"{row['grad_norm']:.6f}"
            )

    with log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                log_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            log_rows
        )

    save_checkpoint_bundle(
        save_dir=final_dir,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    inference = (
        reload_and_inference_smoke(
            save_dir=final_dir,
            data=data,
            device=device,
            seed=20261099,
        )
    )

    files = sorted(
        str(
            path.relative_to(
                final_dir
            )
        )
        for path in final_dir.rglob(
            "*"
        )
        if path.is_file()
    )

    required_files = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }

    missing_required = (
        required_files
        - set(files)
    )

    if missing_required:
        raise RuntimeError(
            "Checkpoint bundle missing: "
            f"{sorted(missing_required)}"
        )

    metadata = {
        "experiment_id":
            protocol[
                "experiment_id"
            ],
        "target_mode":
            mode,
        "target_key":
            target_key,
        "source_git_commit":
            git_commit(),
        "protocol_path":
            str(
                protocol_path
            ),
        "protocol_sha256":
            sha256_file(
                protocol_path
            ),
        "training_data":
            str(
                data_path
            ),
        "training_data_sha256":
            sha256_file(
                data_path
            ),
        "base_checkpoint":
            str(
                base_checkpoint
            ),
        "steps":
            steps,
        "production_steps":
            configured_steps,
        "smoke":
            bool(
                args.smoke
            ),
        "batch_size":
            batch_size,
        "sample_exposures":
            steps
            * batch_size,
        "learning_rate":
            learning_rate,
        "scheduler":
            "constant",
        "optimizer":
            "AdamW",
        "training_seed":
            training_seed,
        "step_seed_rule":
            "training_seed + zero_based_step",
        "total_parameters":
            total_params,
        "trainable_parameters":
            trainable_params,
        "first_loss":
            log_rows[0][
                "loss"
            ],
        "final_loss":
            log_rows[-1][
                "loss"
            ],
        "final_grad_norm":
            log_rows[-1][
                "grad_norm"
            ],
        "checkpoint_files":
            files,
        "reload_inference_shape":
            list(
                inference.shape
            ),
        "reload_inference_finite":
            bool(
                np.isfinite(
                    inference
                ).all()
            ),
        "heldout_minus5_used":
            False,
    }

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

        f.write("\n")

    print()
    print(
        "checkpoint:",
        final_dir,
    )

    print(
        "checkpoint files:"
    )

    for path in files:
        print(
            " ",
            path,
        )

    print()
    print(
        "reload inference:",
        inference.shape,
        "finite=",
        bool(
            np.isfinite(
                inference
            ).all()
        ),
    )

    print()
    print(
        "SMOLVLA LPRD FINETUNE PASSED"
    )


if __name__ == "__main__":
    main()
