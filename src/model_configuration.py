"""Definition of the ModelConfiguration class."""

import os
from dataclasses import dataclass
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

    :param folder: Path to a folder containing files "measurements.csv",
    "stoichiometry.csv" and "priors.csv"

    :param sample_kwargs: dictionary of keyword arguments to
    cmdstanpy.CmdStanModel.sample.
    
    :param likelihood: take measurements into account

    :param run: run this config with the script `fit_all_model_configurations.py`

    """

    name: str
    stan_file: str
    data_folder: str
    sample_kwargs: Dict
    likelihood: bool
    run: bool = True


def load_model_configuration(path: str) -> ModelConfiguration:
    d = toml.load(path)
    mc = ModelConfiguration(
        name=d["name"],
        stan_file=d["stan_file"],
        data_folder=d["data_folder"],
        likelihood=d["likelihood"],
        sample_kwargs=d["sample_kwargs"],
        run=d["run"]
    )
    validate_model_configuration(mc)
    return mc
    

def validate_model_configuration(mc: ModelConfiguration) -> None:
    assert os.path.exists(mc.stan_file)
    assert os.path.exists(mc.data_folder)
    assert type(mc.name) is str
    assert mc.name != ""
    assert type(mc.likelihood) is bool
    assert type(mc.run) is bool
    
