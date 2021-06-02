"""Functions for fitting models using cmdstanpy."""

import os
from typing import List
import arviz as az
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
import pandas as pd
from .model_configuration import ModelConfiguration
from .pandas_to_cmdstanpy import get_stan_input
from .cmdstanpy_to_arviz import get_infd_kwargs

# Location of this file
HERE = os.path.dirname(os.path.abspath(__file__))

# Where to save files
LOO_DIR = os.path.join(HERE, "..", "results", "loo")
SAMPLES_DIR = os.path.join(HERE, "..", "results", "samples")
INFD_DIR = os.path.join(HERE, "..", "results", "infd")
JSON_DIR = os.path.join(HERE, "..", "results", "input_data_json")


def generate_samples(study_name: str, model_configurations: List[ModelConfiguration]) -> None:
    """Run cmdstanpy.CmdStanModel.sample, do diagnostics and save results.

    :param study_name: a string
    """
    infds = {}
    for model_config in model_configurations:
        fit_name = f"{study_name}-{model_config.name}"
        print(f"Fitting model {fit_name}...")
        loo_file = os.path.join(LOO_DIR, f"loo_{fit_name}.pkl")
        infd_file = os.path.join(INFD_DIR, f"infd_{fit_name}.ncdf")
        json_file = os.path.join(JSON_DIR, f"input_data_{fit_name}.json")
        priors = pd.read_csv(os.path.join(model_config.data_folder, "priors.csv"))
        measurements = pd.read_csv(os.path.join(model_config.data_folder, "measurements.csv"))
        S = pd.read_csv(os.path.join(model_config.data_folder, "stoichiometry.csv"), index_col="metabolite")
        likelihood = model_config.likelihood
        stan_input = get_stan_input(measurements, S, priors, likelihood)
        print(f"Writing input data to {json_file}")
        jsondump(json_file, stan_input)
        model = CmdStanModel(
            model_name=fit_name, stan_file=model_config.stan_file
        )
        print(f"Writing csv files to {SAMPLES_DIR}...")
        mcmc = model.sample(
            data=stan_input,
            output_dir=SAMPLES_DIR,
            **model_config.sample_kwargs,
        )
        print(mcmc.diagnose().replace("\n\n", "\n"))
        infd = az.from_cmdstanpy(
            mcmc, **get_infd_kwargs(S, measurements, model_config.sample_kwargs)
        )
        print(az.summary(infd))
        infds[fit_name] = infd
        print(f"Writing inference data to {infd_file}")
        infd.to_netcdf(infd_file)
        print(f"Writing psis-loo results to {loo_file}\n")
        az.loo(infd, pointwise=True).to_pickle(loo_file)
    if len(infds) > 1:
        comparison = az.compare(infds)
        print(f"Loo comparison:\n{comparison}")
        comparison.to_csv(os.path.join(LOO_DIR, "loo_comparison.csv"))
