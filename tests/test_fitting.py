from pathlib import Path

import pandas as pd
import pytest

import arviz as az

from src.fitting import run_stan
from src.model_configuration import load_model_configuration
from src.pandas_to_cmdstanpy import get_coords
from .model_setup import model_small, model_small_rankdef, model_small_rankdef_thermo


@pytest.mark.usefixtures("model_small")
def test_steady_state():
    """
    Test that the results coming from the model satisfy steady state.
    """
    config = load_model_configuration("test_small_prior.toml")
    S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col=0)
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    priors = pd.read_csv(config.data_folder / "priors.csv")
    # This will be cleaned up with the rest of the files
    config.result_dir = config.data_folder / "results"
    # Remove the results file
    mcmc = run_stan(config)
    data = az.from_cmdstanpy(mcmc, coords=get_coords(S, measurements, priors, config.order))
    df = data.posterior.flux.to_dataframe().unstack("flux_dim_0")
    # Check the steady state for each sample
    for i in range(df.shape[0]):
        assert pytest.approx(0, abs=1e-6) == S @ df.iloc[i, :].values, "All flux samples should be balanced"

@pytest.mark.usefixtures("model_small_rankdef")
def test_rank_deficient():
    """Test fitting with a rank-deficient matrix"""
    config = load_model_configuration("test_small_prior.toml")
    S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col=0)
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    priors = pd.read_csv(config.data_folder / "priors.csv")
    # This will be cleaned up with the rest of the files
    config.result_dir = config.data_folder / "results"
    # Remove the results file
    mcmc = run_stan(config)
    data = az.from_cmdstanpy(mcmc, coords=get_coords(S, measurements, priors, config.order))
    df = data.posterior.flux.to_dataframe().unstack("flux_dim_0")
    # Check the steady state for each sample
    for i in range(df.shape[0]):
        assert pytest.approx(0, abs=1e-6) == S @ df.iloc[i, :].values, "All flux samples should be balanced"

# THe test case doesn't actually produce a reduced rank covariance matrix
# @pytest.mark.usefixtures("model_small_rankdef_thermo")
# def test_rank_deficient():
#     """Test fitting with a rank-deficient matrix"""
#     config = load_model_configuration("test_small_prior.toml")
#     S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col=0)
#     measurements = pd.read_csv(config.data_folder / "measurements.csv")
#     priors = pd.read_csv(config.data_folder / "priors.csv")
#     # This will be cleaned up with the rest of the files
#     config.result_dir = config.data_folder / "results"
#     # Remove the results file
#     mcmc = run_stan(config)
#     data = az.from_cmdstanpy(mcmc, coords=get_coords(S, measurements, priors, config.order))
#     df = data.posterior.flux.to_dataframe().unstack("flux_dim_0")
#     # Check the steady state for each sample
#     for i in range(df.shape[0]):
#         assert pytest.approx(0, abs=1e-6) == S @ df.iloc[i, :].values, "All flux samples should be balanced"