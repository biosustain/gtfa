"""Functions for fitting models using cmdstanpy."""
import itertools
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List
import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
import pandas as pd
from src.model_configuration import ModelConfiguration
from src.pandas_to_cmdstanpy import get_stan_input
from src.cmdstanpy_to_arviz import get_infd_kwargs


logger = logging.getLogger(__name__)


def stan_input_from_dir(data_folder: Path, order=None, likelihood=False):
    priors = pd.read_csv(data_folder / "priors.csv")
    priors_cov = pd.read_csv(data_folder / "priors_cov.csv", index_col=0)
    measurements = pd.read_csv(data_folder / "measurements.csv")
    S = pd.read_csv(data_folder / "stoichiometry.csv", index_col="metabolite")
    return get_stan_input(measurements, S, priors, priors_cov, likelihood, order)


def stan_input_from_config(config: ModelConfiguration):
    return stan_input_from_dir(config.data_folder, config.order, config.likelihood)

def generate_samples(config: ModelConfiguration) -> None:
    """Run cmdstanpy.CmdStanModel.sample, do diagnostics and save results.

    :param study_name: a string
    """
    logger.info(f"Fitting model configuration {config.name}...")
    infd_file = config.result_dir / "infd.nc"
    json_file = config.result_dir / "input_data.json"
    stan_input = stan_input_from_config(config)
    logger.info(f"Writing input data to {json_file}")
    jsondump(str(json_file), stan_input)
    mcmc = run_stan(config)
    # Write the files
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col="metabolite")
    write_files(S, config, infd_file, mcmc, measurements)


def run_stan(config):
    stan_input = stan_input_from_config(config)
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
    return mcmc


def write_files(S, config, infd_file, mcmc, measurements):
    logger.info(mcmc.diagnose().replace("\n\n", "\n"))
    infd = get_infd(S, config, mcmc, measurements)
    logger.info(az.summary(infd))
    logger.info(f"Writing inference data to {infd_file}")
    infd.to_netcdf(str(infd_file))


def get_infd(S, config, mcmc, measurements):
    infd_kwargs = get_infd_kwargs(S, measurements, config.order, config.sample_kwargs)
    infd = az.from_cmdstan(
        mcmc.runset.csv_files, **infd_kwargs
    )
    return infd
