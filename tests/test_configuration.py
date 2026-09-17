from pathlib import Path

import pytest

from so101_vla_response.configuration import ConfigError, load_experiment_bundle, load_yaml


def test_response_comparison_config_bundle_loads() -> None:
    bundle = load_experiment_bundle(Path("configs/experiment/response_comparison.yaml"))

    assert bundle.experiment["id"] == "response_comparison_v0"
    assert set(bundle.sections) == {"dataset", "robot", "scene", "task"}
    assert set(bundle.policy_sections) == {"pi05", "smolvla"}
    assert set(bundle.training_sections) == {"pi05", "smolvla"}

    dataset = bundle.section("dataset").root
    assert dataset["recording_hz"] == 30
    assert dataset["episode_duration_s"] == 7.0
    assert dataset["frames_per_episode"] == 210
    assert dataset["object_offsets_m"]["training"] == [-0.02, 0.0, 0.02]


def test_load_yaml_requires_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "not_mapping.yaml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="document root"):
        load_yaml(config_path)
