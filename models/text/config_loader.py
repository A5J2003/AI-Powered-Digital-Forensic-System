import json
import os

def load_config(config_path=None):
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "models", "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config