import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.analysis.analyse_results import analyse
from src.fitting import generate_samples
from src.model_configuration import load_model_configuration
from src.model_conversion import write_gollub2020_models

RESULTS_DIR = Path(__file__).parent / "results"
logger = logging.getLogger()

def setup_logging(config):
    FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    HANDLERS = [logging.FileHandler(config.result_dir / "out.log"), logging.StreamHandler()]
    if config.devel & config.verbose:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=HANDLERS)
    elif config.verbose | config.devel:
        logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=HANDLERS)
    else:
        logging.basicConfig(level=logging.WARNING, format=FORMAT, handlers=HANDLERS)
    #


def check_input(config_files):
    if not all(Path(p).exists() for p in config_files):
        logger.critical("All arguments should be valid paths to config files")
        sys.exit(2)
    if len(config_files) == 0:
        logger.critical("At least one config file should be given")
        sys.exit(2)


def run_config(config):
    # If this config is for development only then clear out any previous results
    this_results_dir = RESULTS_DIR / config.name
    if config.devel:
        if this_results_dir.exists():
            shutil.rmtree(this_results_dir)
    else:
        # Make sure the parent directory exists for the first run
        if not this_results_dir.exists():
            this_results_dir.mkdir()
        # Experimental configurations store all runs with the datetime of the run
        this_results_dir = this_results_dir / datetime.now().strftime("%Y%m%d%H%M%S")
    this_results_dir.mkdir()
    config.result_dir = this_results_dir
    # Set up logging
    setup_logging(config)
    generate_samples(config)
    # Generate the analysis files
    if config.analyse:
        analyse(config)



if __name__ == "__main__":
    print("RUNNING")
    logging.getLogger().setLevel(logging.CRITICAL)
    gollub_files = list((Path(__file__).parent / "data" / "raw" / "from_gollub_2020").glob("**/*.mat"))
    assert len(gollub_files) > 0
    temp_dir = Path("temp_dir")
    if not temp_dir.exists():
        temp_dir.mkdir()
    write_gollub2020_models(gollub_files[:1], temp_dir)




    # # Add the script directory to the path
    # sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    # # For now all arguments should just be valid paths
    # args = sys.argv[1:]
    # check_input(args)
    # configs = [load_model_configuration(filename) for filename in args]
    # for config in configs:
    #     run_config(config)
