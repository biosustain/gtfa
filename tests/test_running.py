import os
import shutil
from pathlib import Path

import pytest

from gtfa import run_config
from src.model_configuration import load_model_configuration


def test_run_direct(model_small):
    """ Run the command directly"""
    # Clean up if the results dir is still there
    if Path("../results/small_test").exists():
        shutil.rmtree("../results/small_test")
    run_config(load_model_configuration("test_small_prior.toml"))
    assert Path("../results/small_test").exists()
    assert len(list(Path("../results/small_test").iterdir())) == 1, "There should be a single folder in there"
    shutil.rmtree("../results/small_test")


def test_run_cmdline(model_small):
    """ Test running gtfa from the command line"""
    # Clean up if the results dir is still there
    if Path("../results/small_test").exists():
        shutil.rmtree("../results/small_test")
    os.system("python ../gtfa.py test_small_prior.toml")
    # The results should be written to the results directory
    assert Path("../results/small_test").exists()
    assert len(list(Path("../results/small_test").iterdir())) == 1, "There should be a single folder in there"
    shutil.rmtree("../results/small_test")


def test_run_cmdline_many(model_small):
    """ Test running gtfa from the command line with multiple runs"""
    # Clean up if the results dir is still there
    if Path("../results/small_test").exists():
        shutil.rmtree("../results/small_test")
    # Execute process
    os.system("python ../gtfa.py test_small_prior.toml")
    os.system("python ../gtfa.py test_small_prior.toml")
    os.system("python ../gtfa.py test_small_prior.toml")
    os.system("python ../gtfa.py test_small_prior.toml")
    os.system("python ../gtfa.py test_small_prior.toml")
    # The results should be written to the results directory
    assert Path("../results/small_test").exists()
    # There should be five child directories
    assert len(list(Path("../results/small_test").iterdir())) == 5
    shutil.rmtree("../results/small_test")
