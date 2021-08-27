import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.analysis.analyse_results import pair_plots
from src.fitting import generate_samples
from src.model_configuration import load_model_configuration

RESULTS_DIR = Path("results")


def check_bad_input(args):
    failed = False
    if not all(Path(p).exists() for p in config_files):
        print("All arguments should be valid config files")
        return True
    if len(config_files) == 0:
        print("At least one config file should be given")
        return True
    return failed


def run_configs(args):
    if check_bad_input(args):
        return
    configs = [load_model_configuration(filename) for filename in args]
    for config in configs:
        # If this config is for development only then clear out any previous results
        this_results_dir = RESULTS_DIR / config.name
        if config.devel:
            if this_results_dir.exists():
                shutil.rmtree(this_results_dir)
        else:
            # Experimental configurations store all runs with the datetime of the run
            this_results_dir = this_results_dir / datetime.now().strftime("%Y%m%d%H%M%S")
        this_results_dir.mkdir()
        config.result_dir = this_results_dir
        generate_samples(config)
        # Generate the analysis files
        if config.analyse:
            # If we are developing we don't want to save the pair plot files
            save_plots = not config.devel
            pair_plots(config, save=save_plots)

if __name__ == "__main__":
    # For now all arguments should just be valid paths
    config_files = sys.argv[1:]
    run_configs(config_files)
