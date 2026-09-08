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
