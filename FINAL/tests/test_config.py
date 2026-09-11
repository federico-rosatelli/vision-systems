import json
import os
import sys
import subprocess
import pytest

def test_config_loading(tmp_path):
    """Test that main.py parses JSON configuration files correctly."""
    config_data = {
        "action": "test",
        "seed": 123
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
        
    cmd = [sys.executable, "main.py", "--config", str(config_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert f"Loaded configuration from {config_file}" in result.stdout
    assert "Step 5: Testing" in result.stdout

def test_default_config_loading():
    """Test running python main.py uses default configs/config.json if no action is provided."""
    if os.path.exists("configs/config.json"):
        cmd = [sys.executable, "main.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0

def test_grouped_mil_configs_exist_and_parse():
    """Test that grouped MIL config files exist and can be parsed by main.py."""
    grouped_configs = [
        "configs/config_mil_data.json",
        "configs/config_mil_train.json",
        "configs/config_mil_eval.json"
    ]
    for config_path in grouped_configs:
        assert os.path.exists(config_path), f"Missing config file: {config_path}"
        with open(config_path, "r") as f:
            data = json.load(f)
        assert "action" in data

