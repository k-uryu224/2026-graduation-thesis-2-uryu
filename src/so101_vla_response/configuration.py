"""Configuration loading helpers for experiment YAML files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in misconfigured envs
    yaml = None


class ConfigError(ValueError):
    """Raised when a configuration file has an invalid shape."""


@dataclass(frozen=True)
class ConfigSection:
    """A loaded config file and its expected top-level section."""

    kind: str
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> dict[str, Any]:
        return _expect_mapping(self.data, self.kind, self.path)


@dataclass(frozen=True)
class ExperimentBundle:
    """An experiment config with all referenced config files resolved."""

    path: Path
    experiment: dict[str, Any]
    sections: dict[str, ConfigSection]
    policy_sections: dict[str, ConfigSection]
    training_sections: dict[str, ConfigSection]

    def section(self, name: str) -> ConfigSection:
        try:
            return self.sections[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sections))
            raise ConfigError(f"unknown config section {name!r}; available: {available}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc

    if yaml is not None:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    else:
        raw = _parse_repo_yaml_subset(text, config_path)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the document root")
    return raw


def load_section(path: str | Path, kind: str) -> ConfigSection:
    """Load one config file and validate its top-level section key."""

    config_path = Path(path)
    data = load_yaml(config_path)
    root = _expect_mapping(data, kind, config_path)
    identifier = root.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ConfigError(f"{config_path}: {kind}.id must be a non-empty string")
    return ConfigSection(kind=kind, path=config_path, data=data)


def load_experiment_bundle(path: str | Path) -> ExperimentBundle:
    """Load an experiment config and all config files referenced by it."""

    experiment_path = Path(path)
    data = load_yaml(experiment_path)
    experiment = _expect_mapping(data, "experiment", experiment_path)
    if not isinstance(experiment.get("id"), str) or not experiment["id"]:
        raise ConfigError(f"{experiment_path}: experiment.id must be a non-empty string")

    refs = _expect_mapping(data, "config_refs", experiment_path)
    base_dir = experiment_path.parent

    section_kinds = {
        "robot": "robot",
        "scene": "scene",
        "task": "task",
        "dataset": "dataset",
    }
    sections = {
        name: load_section(_resolve_ref(base_dir, refs, name), kind)
        for name, kind in section_kinds.items()
    }

    policy_sections = _load_named_ref_sections(
        base_dir=base_dir,
        refs=refs,
        ref_name="policies",
        kind="policy",
        experiment_path=experiment_path,
    )
    training_sections = _load_named_ref_sections(
        base_dir=base_dir,
        refs=refs,
        ref_name="training",
        kind="training",
        experiment_path=experiment_path,
    )

    return ExperimentBundle(
        path=experiment_path,
        experiment=experiment,
        sections=sections,
        policy_sections=policy_sections,
        training_sections=training_sections,
    )


def _load_named_ref_sections(
    *,
    base_dir: Path,
    refs: dict[str, Any],
    ref_name: str,
    kind: str,
    experiment_path: Path,
) -> dict[str, ConfigSection]:
    named_refs = _expect_mapping(refs, ref_name, experiment_path)
    sections = {}
    for name, value in named_refs.items():
        if not isinstance(value, str):
            raise ConfigError(f"{experiment_path}: config_refs.{ref_name}.{name} must be a path")
        sections[name] = load_section((base_dir / value).resolve(), kind)
    return sections


def _resolve_ref(base_dir: Path, refs: dict[str, Any], name: str) -> Path:
    value = refs.get(name)
    if not isinstance(value, str):
        raise ConfigError(f"config_refs.{name} must be a path")
    return (base_dir / value).resolve()


def _expect_mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {key} must be a mapping")
    return value


def _parse_repo_yaml_subset(text: str, path: Path) -> Any:
    """Parse the small YAML subset used by this repository's configs.

    PyYAML is preferred when available. This fallback keeps config loading usable
    in minimal environments, but it intentionally supports only mappings,
    sequences of scalar values, and scalar values used in `configs/`.
    """

    tokens = _tokenize_yaml_subset(text, path)
    if not tokens:
        return None
    value, index = _parse_block(tokens, 0, tokens[0][0], path)
    if index != len(tokens):
        line_no = tokens[index][2]
        raise ConfigError(f"{path}:{line_no}: unexpected trailing YAML content")
    return value


def _tokenize_yaml_subset(text: str, path: Path) -> list[tuple[int, str, int]]:
    tokens = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            raise ConfigError(f"{path}:{line_no}: tabs are not supported in YAML indentation")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        tokens.append((indent, raw_line.strip(), line_no))
    return tokens


def _parse_block(
    tokens: list[tuple[int, str, int]],
    index: int,
    indent: int,
    path: Path,
) -> tuple[Any, int]:
    if tokens[index][1].startswith("- "):
        return _parse_sequence(tokens, index, indent, path)
    return _parse_mapping(tokens, index, indent, path)


def _parse_mapping(
    tokens: list[tuple[int, str, int]],
    index: int,
    indent: int,
    path: Path,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(tokens):
        current_indent, content, line_no = tokens[index]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ConfigError(f"{path}:{line_no}: unexpected indentation")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ConfigError(f"{path}:{line_no}: expected 'key: value'")

        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ConfigError(f"{path}:{line_no}: empty mapping key")

        index += 1
        if raw_value:
            mapping[key] = _parse_scalar(raw_value)
            continue

        if index < len(tokens) and tokens[index][0] > current_indent:
            mapping[key], index = _parse_block(tokens, index, tokens[index][0], path)
        else:
            mapping[key] = {}
    return mapping, index


def _parse_sequence(
    tokens: list[tuple[int, str, int]],
    index: int,
    indent: int,
    path: Path,
) -> tuple[list[Any], int]:
    values = []
    while index < len(tokens):
        current_indent, content, line_no = tokens[index]
        if current_indent < indent:
            break
        if current_indent != indent or not content.startswith("- "):
            break

        raw_value = content[2:].strip()
        index += 1
        if raw_value:
            values.append(_parse_scalar(raw_value))
            continue
        if index >= len(tokens) or tokens[index][0] <= current_indent:
            values.append(None)
            continue
        value, index = _parse_block(tokens, index, tokens[index][0], path)
        values.append(value)

    if not values:
        line_no = tokens[index][2] if index < len(tokens) else "EOF"
        raise ConfigError(f"{path}:{line_no}: empty sequence")
    return values, index


def _parse_scalar(value: str) -> Any:
    if value in {"null", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "[]":
        return []
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+",
        value,
    ):
        return float(value)
    return value
