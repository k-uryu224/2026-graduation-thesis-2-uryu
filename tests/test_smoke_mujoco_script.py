import importlib.util
from pathlib import Path


def _load_smoke_model_xml() -> str:
    script_path = Path("scripts/smoke_mujoco.py")
    spec = importlib.util.spec_from_file_location("smoke_mujoco", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SMOKE_MODEL_XML


def test_smoke_mujoco_xml_contains_basic_objects() -> None:
    smoke_model_xml = _load_smoke_model_xml()

    assert '<mujoco model="smoke_test">' in smoke_model_xml
    assert 'name="box_freejoint"' in smoke_model_xml
    assert 'name="box_center"' in smoke_model_xml
    assert 'name="overview"' in smoke_model_xml
