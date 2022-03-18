from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fake_data_generation import generate_data_and_config, switch_config, generate_data, \
    DEFAULT_ENZYME_MEASUREMENT_ERROR, DEFAULT_MET_MEASUREMENT_ERROR, DEFAULT_FLUX_MEASUREMENT_ERROR
from src.fitting import stan_input_from_config
from src.model_configuration import ModelConfiguration, load_model_configuration

base_dir = Path(__file__).parent.parent


# NOTE: This could be streamlined with a fixture later
# @pytest.mark.parametrize("model", ["model_small", "ecoli_model", "small_model_irreversible"])

def test_small_model(model_small):
    SAMPLES_PER_PARAM = 100
    NUM_CONDITIONS = 5
    MEASURED_PARAMS = ["all"]
    np.random.seed(0)
    # Take base config
    base_config = load_model_configuration(base_dir / "tests" / "test_small_likelihood.toml")
    # Generate fake data
    new_config = switch_config(base_config, MEASURED_PARAMS, num_conditions=NUM_CONDITIONS,
                               samples_per_param=SAMPLES_PER_PARAM)
    # Load the datafiles
    S = pd.read_csv(new_config.data_folder / "stoichiometry.csv", index_col=0)
    # Generate the data
    # Load the stan input with the original data (we won't use any of the measurements, just the priors)
    stan_input = stan_input_from_config(new_config)
    data_df = generate_data(stan_input, S, SAMPLES_PER_PARAM, NUM_CONDITIONS)
    # Check the means and sd of the generated data match their parameters
    # NOTE: This could probably be done with a clever groupby
    # Find the different parameter groups
    matches = (data_df["P"].shift().values == data_df["P"].values).all(axis=1)
    groups = (~matches).cumsum() - 1
    single_params = data_df.loc[~matches, "P"]
    for i in range(max(groups)):
        group_indices = groups == i
        group_means = data_df["M"].loc[group_indices].mean()
        group_sd = data_df["M"].loc[group_indices].std()
        assert np.allclose(group_means["enzyme"], single_params.iloc[i]["enzyme"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["flux"], single_params.iloc[i]["flux"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["mic"], single_params.iloc[i]["mic"], atol=5e-1, rtol=5e-1)
        # Now the sds
        assert np.allclose(group_sd["enzyme"], DEFAULT_ENZYME_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["flux"], DEFAULT_FLUX_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["mic"], DEFAULT_MET_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)


@pytest.mark.xfail(expected=NotImplementedError, reason="Duplicate compounds aren't supported yet")
def test_ecoli_model(ecoli_model):
    SAMPLES_PER_PARAM = 100
    NUM_CONDITIONS = 5
    MEASURED_PARAMS = ["all"]
    np.random.seed(0)
    # Take base config
    base_config = load_model_configuration(base_dir / "tests" / "test_small_likelihood.toml")
    # Generate fake data
    new_config = switch_config(base_config, MEASURED_PARAMS, num_conditions=NUM_CONDITIONS,
                               samples_per_param=SAMPLES_PER_PARAM)
    # Load the datafiles
    S = pd.read_csv(new_config.data_folder / "stoichiometry.csv", index_col=0)
    # Generate the data
    # Load the stan input with the original data (we won't use any of the measurements, just the priors)
    stan_input = stan_input_from_config(new_config)
    data_df = generate_data(stan_input, S, SAMPLES_PER_PARAM, NUM_CONDITIONS)
    # Check the means and sd of the generated data match their parameters
    # NOTE: This could probably be done with a clever groupby
    # Find the different parameter groups
    matches = (data_df["P"].shift().values == data_df["P"].values).all(axis=1)
    groups = (~matches).cumsum() - 1
    single_params = data_df.loc[~matches, "P"]
    for i in range(max(groups)):
        group_indices = groups == i
        group_means = data_df["M"].loc[group_indices].mean()
        group_sd = data_df["M"].loc[group_indices].std()
        assert np.allclose(group_means["enzyme"], single_params.iloc[i]["enzyme"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["flux"], single_params.iloc[i]["flux"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["mic"], single_params.iloc[i]["mic"], atol=5e-1, rtol=5e-1)
        # Now the sds
        assert np.allclose(group_sd["enzyme"], DEFAULT_ENZYME_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["flux"], DEFAULT_FLUX_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["mic"], DEFAULT_MET_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)


def test_small_model_rankdef(model_small_rankdef_thermo):
    SAMPLES_PER_PARAM = 100
    NUM_CONDITIONS = 5
    MEASURED_PARAMS = ["all"]
    np.random.seed(0)
    # Take base config
    base_config = load_model_configuration(base_dir / "tests" / "test_small_likelihood.toml")
    # Generate fake data
    new_config = switch_config(base_config, MEASURED_PARAMS, num_conditions=NUM_CONDITIONS,
                               samples_per_param=SAMPLES_PER_PARAM)
    # Load the datafiles
    S = pd.read_csv(new_config.data_folder / "stoichiometry.csv", index_col=0)
    # Generate the data
    # Load the stan input with the original data (we won't use any of the measurements, just the priors)
    stan_input = stan_input_from_config(new_config)
    data_df = generate_data(stan_input, S, SAMPLES_PER_PARAM, NUM_CONDITIONS)
    # Check the means and sd of the generated data match their parameters
    # NOTE: This could probably be done with a clever groupby
    # Find the different parameter groups
    matches = (data_df["P"].shift().values == data_df["P"].values).all(axis=1)
    groups = (~matches).cumsum() - 1
    single_params = data_df.loc[~matches, "P"]
    for i in range(max(groups)):
        group_indices = groups == i
        group_means = data_df["M"].loc[group_indices].mean()
        group_sd = data_df["M"].loc[group_indices].std()
        assert np.allclose(group_means["enzyme"], single_params.iloc[i]["enzyme"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["flux"], single_params.iloc[i]["flux"], atol=5e-1, rtol=5e-1)
        assert np.allclose(group_means["mic"], single_params.iloc[i]["mic"], atol=5e-1, rtol=5e-1)
        # Now the sds
        assert np.allclose(group_sd["enzyme"], DEFAULT_ENZYME_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["flux"], DEFAULT_FLUX_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
        assert np.allclose(group_sd["mic"], DEFAULT_MET_MEASUREMENT_ERROR, atol=5e-2, rtol=5e-2)
