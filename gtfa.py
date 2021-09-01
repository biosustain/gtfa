import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.analysis.analyse_results import pair_plots
from src.fitting import generate_samples
from src.model_configuration import load_model_configuration

RESULTS_DIR = Path("results")
logger = logging.getLogger()

def setup_logging(config):
    FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    HANDLERS = [logging.FileHandler(config.result_dir / "out.log"), logging.StreamHandler()]
    if config.devel:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=HANDLERS)
    elif config.verbose:
        logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=HANDLERS)
    else:
        logging.basicConfig(level=logging.WARNING, format=FORMAT, handlers=HANDLERS)
    #


def check_input(config_files):
    if not all(Path(p).exists() for p in config_files):
        logger.critical("All arguments should be valid config files")
        sys.exit(2)
    if len(config_files) == 0:
        logger.critical("At least one config file should be given")
        sys.exit(2)


def run_configs(args):
    check_input(args)
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
        # Set up logging
        setup_logging(config)
        print(config.devel)
        logger.debug("TEST")
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
