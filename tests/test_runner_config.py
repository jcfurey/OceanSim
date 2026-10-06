"""Runner configuration checks without starting Isaac Sim."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "isaacsim/oceansim/standalone/oceansim_ros2.py"
spec = importlib.util.spec_from_file_location("oceansim_runner_config", PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_demo_camera_resolution_and_step_limit_overrides(tmp_path):
    config = tmp_path / "demo.json"
    config.write_text(json.dumps({"camera_resolution": [960, 540], "max_steps": 120}))
    cfg = runner.load_config(runner.parse_args([
        "--config", str(config), "--max-steps", "60", "--camera-resolution", "640", "360",
    ]))
    assert cfg["max_steps"] == 60
    assert cfg["camera_resolution"] == [640, 360]
    assert cfg["sensors"]["sonar"]  # partial config keeps the sensor defaults


@pytest.mark.parametrize("steps", [0, -2])
def test_cli_requires_positive_smoke_step_limit(steps):
    with pytest.raises(ValueError, match="--max-steps must be positive"):
        runner.load_config(runner.parse_args(["--max-steps", str(steps)]))


@pytest.mark.parametrize("resolution", [[0, 540], [-1, 540], [960], [960.5, 540]])
def test_invalid_camera_resolution_fails_before_gpu_start(tmp_path, resolution):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"camera_resolution": resolution}))
    with pytest.raises(ValueError, match="camera_resolution"):
        runner.load_config(runner.parse_args(["--config", str(config)]))
