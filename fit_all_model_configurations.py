"""Fit the models in MODEL_CONFIGURATIONS to real data."""

from datetime import datetime
import os
import pandas as pd
from src.model_configuration import load_model_configuration

from src.fitting import generate_samples

# only display messages with at least this severity
LOGGER_LEVEL = 40

CONFIG_DIR = "model_configurations"


def main():
    """Run the script."""
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    study_name = f"real_study-{now}"
    config_files = [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".toml")
    ]
    for config_file in config_files:
        model_config = load_model_configuration(config_file)
        generate_samples(model_config)


if __name__ == "__main__":
    main()
