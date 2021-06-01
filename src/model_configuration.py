"""Definition of the ModelConfiguration class."""

from dataclasses import dataclass
from typing import Callable, Dict
import pandas as pd


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

    """

    name: str
    stan_file: str
    data_folder: str
    sample_kwargs: Dict
    likelihood: bool
