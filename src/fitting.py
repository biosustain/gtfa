"""Functions for fitting models using cmdstanpy."""
import itertools
import logging
import os
import re
import warnings
from typing import List
import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
import pandas as pd
from .model_configuration import ModelConfiguration
from .pandas_to_cmdstanpy import get_stan_input
from .cmdstanpy_to_arviz import get_infd_kwargs


logger = logging.getLogger(__name__)


def generate_samples(config: ModelConfiguration) -> None:
    """Run cmdstanpy.CmdStanModel.sample, do diagnostics and save results.

    :param study_name: a string
    """
    logger.info(f"Fitting model configuration {config.name}...")
    infd_file = config.result_dir / "infd.nc"
    json_file = config.result_dir / "input_data.json"
    priors = pd.read_csv(config.data_folder / "priors.csv")
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col="metabolite")
    stan_input = get_stan_input(measurements, S, priors, config.likelihood)
    logger.info(f"Writing input data to {json_file}")
    jsondump(str(json_file), stan_input)
    model = CmdStanModel(
        model_name=config.name, stan_file=str(config.stan_file)
    )
    # Make the samples directory
    sample_dir = config.result_dir / "samples"
    sample_dir.mkdir()
    logger.info(f"Writing csv files to {sample_dir}...")
    mcmc = model.sample(
        data=stan_input,
        output_dir=str(sample_dir),
        **config.sample_kwargs,
    )
    logger.info(mcmc.diagnose().replace("\n\n", "\n"))
    infd_kwargs = get_infd_kwargs(S, measurements, config.sample_kwargs)
    infd = az.from_cmdstan(
        mcmc.runset.csv_files, **infd_kwargs
    )
    logger.info(az.summary(infd))
    logger.info(f"Writing inference data to {infd_file}")
    infd.to_netcdf(str(infd_file))
