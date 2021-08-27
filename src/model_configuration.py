"""Definition of the ModelConfiguration class."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict
import pandas as pd
import toml


@dataclass
class ModelConfiguration:
    """Container for a path to a Stan model and some configuration.

    For example, you may want to compare how well two stan programs fit the
    same data, or how well the same model fits the data with different
    covariates.

    :param name: string name identifying the model configuration

    :param stan_file: Path to a Stan program

    :param data_folder: Path to a folder containing files "measurements.csv",
    "stoichiometry.csv" and "priors.csv"

    :param result_dir: Path to the result files of this run

    :param sample_kwargs: dictionary of keyword arguments to
    cmdstanpy.CmdStanModel.sample.
    
    :param likelihood: take measurements into account

    :param devel: This is being run for development and can overwrite previous values

    :param run: run this config with the script `fit_all_model_configurations.py`

    :param analyse: run analysis functions and save results
    """

    name: str
    stan_file: Path
    data_folder: Path
    sample_kwargs: Dict
    likelihood: bool
    result_dir: Path = Path("empty")
    devel: bool = True
    analyse: bool = False
    disp_plot: bool = True
    save_plot: bool = False


def load_model_configuration(path: str) -> ModelConfiguration:
    d = toml.load(path)
    mc = ModelConfiguration(
        name=d["name"],
        stan_file=Path(d["stan_file"]),
        data_folder=Path(d["data_folder"]),
        likelihood=d["likelihood"],
        sample_kwargs=d["sample_kwargs"],
        analyse=d["analyse"],
        devel=d["devel"]
    )
    validate_model_configuration(mc)
    return mc
    

def validate_model_configuration(mc: ModelConfiguration) -> None:
    assert os.path.exists(mc.stan_file)
    assert os.path.exists(mc.data_folder)
    assert type(mc.name) is str
    assert mc.name != ""
    assert type(mc.likelihood) is bool
    
