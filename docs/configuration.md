# Configuration

## Purpose

The repository is config-driven. Experimental conditions should be visible in
`configs/` before they are used by scripts or source modules.

The current config layer is intentionally small. It records the Phase 1 research
design and leaves asset-dependent values as `null` with an explicit `status`
field until MuJoCo, SO-101 sites, cameras, and model adapters are verified.

## Config categories

| Directory | Top-level key | Purpose |
|---|---|---|
| `configs/robot/` | `robot` | SO-101 asset, joints, control rate, TCP site |
| `configs/scene/` | `scene` | MuJoCo world, table, object, cameras, perturbation |
| `configs/task/` | `task` | instruction, phase order, success condition |
| `configs/dataset/` | `dataset` | recording frequency, episode length, variations, fields |
| `configs/policy/` | `policy` | model family, checkpoint, adapters, common interface |
| `configs/training/` | `training` | training method and model-specific hyperparameters |
| `configs/experiment/` | `experiment` | one experiment's referenced configs and comparisons |

## Unit rules

| Quantity | Unit | Key naming rule |
|---|---|---|
| Position / displacement | m | include `_m` for scalar or vector values |
| Time / duration | s | include `_s` for scalar values |
| Frequency | Hz | include `_hz` |
| Orientation | quaternion | include ordering such as `_xyzw` |
| Joint angle | rad | record in adapter or key name before use |

## Experiment references

An experiment config is the entry point for a runnable setup. It points to the
other config files instead of duplicating their contents:

```yaml
config_refs:
  robot: ../robot/so101.yaml
  scene: ../scene/top_grasp.yaml
  task: ../task/pick_object.yaml
  dataset: ../dataset/grasp_dataset.yaml
  policies:
    pi05: ../policy/pi05.yaml
    smolvla: ../policy/smolvla.yaml
```

Code should load `configs/experiment/response_comparison.yaml` first, then
resolve its references. This keeps each experiment reproducible while allowing
shared robot, scene, dataset, and policy configs.

## Draft values

The following values are part of the current design and are safe to use in early
implementation:

| Field | Value |
|---|---|
| `recording_hz` | `30` |
| `episode_duration_s` | `7.0` |
| `frames_per_episode` | `210` |
| `training_offsets_m` | `[-0.02, 0.0, 0.02]` |
| `training_variations_per_offset` | `20` |
| `holdout_variations` | `5` |

The following values must remain uncommitted until verified:

| Field | Why |
|---|---|
| MuJoCo model XML path | SO-101 asset has not been selected in this repo |
| `world` axis for perturbation | viewer and model orientation must be checked |
| table height | depends on the MuJoCo scene geometry |
| object baseline pose | depends on the final scene coordinate system |
| camera names and extrinsics | depend on the final scene model |
| policy checkpoints and adapters | depend on the selected pi05 / SmolVLA setup |

## Loading in Python

Use `so101_vla_response.configuration.load_experiment_bundle`:

```python
from pathlib import Path

from so101_vla_response.configuration import load_experiment_bundle

bundle = load_experiment_bundle(Path("configs/experiment/response_comparison.yaml"))
dataset = bundle.section("dataset").root
print(dataset["recording_hz"])
```

The loader validates only the config shape and references at this stage. It does
not validate MuJoCo-specific values that are still intentionally unknown.

If PyYAML is installed, the loader uses it. In minimal environments, it falls
back to the small YAML subset used by this repository's config files.
