import logging
import shutil
from pathlib import Path

import scipy
from cobra.util.array import create_stoichiometric_matrix
import cobra.util.array
import numpy as np
import pandas as pd
import pytest

from src.dgf_estimation import calc_model_dgfs_with_prediction_error
from src.fitting import generate_samples, stan_input_from_config
from src.model_configuration import load_model_configuration
from src.model_conversion import write_gollub2020_models, get_compartment_conditions

# Don't delete
# from .model_setup import ecoli_model, model_small
from .model_setup import ecoli_model, model_small, temp_dir
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

base_dir = Path(__file__).parent.parent

@pytest.mark.usefixtures("ecoli_model")
def test_model_writing(ecoli_model):
    temp_dir = Path("temp_dir")
    # Check the stoichiometry of a few reactions at random
    S = pd.read_csv(temp_dir / "stoichiometry.csv", index_col="metabolite")
    model_s = create_stoichiometric_matrix(ecoli_model)
    assert ecoli_model.reactions[0].id == "PFK"
    assert all(model_s[:, 0] == S["PFK"])
    assert ecoli_model.reactions[70].id == 'FUMt2_2'
    assert all(model_s[:, 70] == S['FUMt2_2'])
    # Check the dgf priors
    priors = pd.read_csv(temp_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs_with_prediction_error(ecoli_model)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(temp_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())


@pytest.mark.usefixtures("model_small")
def test_model_writing_small(model_small):
    # Add the test dir
    temp_dir = Path("temp_dir")
    # Check the dgf priors
    priors = pd.read_csv(temp_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs_with_prediction_error(model_small)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(temp_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())


@pytest.mark.usefixtures("model_small")
def test_small_model_prior(model_small):
    # Add the test dir
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    # Make a config and run this test model
    config = load_model_configuration("test_small_prior.toml")
    config.result_dir = result_dir
    # Run the sampling
    generate_samples(config)
    # Check results files
    priors = pd.read_csv(temp_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs_with_prediction_error(model_small)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(temp_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())


@pytest.mark.usefixtures("temp_dir")
def test_gollub_files_read_singles(temp_dir):
    """ Test that all gollub model files can be read and converted individually"""
    gollub_files = list((temp_dir.parent.parent / "data" / "raw" / "from_gollub_2020").glob("**/*.mat"))
    assert len(gollub_files) > 0
    # Choose two files at random
    np.random.seed(42)
    gollub_files = np.random.choice(gollub_files, 2, replace=False)
    for f in gollub_files:
        write_gollub2020_models([f], temp_dir)
        # Load the true data
        model_struct = scipy.io.loadmat(f)
        model = cobra.io.mat.from_mat_struct(model_struct["model"])
        # Add the conditions
        model.compartment_conditions = get_compartment_conditions(model, model_struct)
        # Add the excluded reactions
        exclude_rxns = model_struct["model"]["isConstraintRxn"][0, 0].flatten() == 0
        model.Exclude_list = [model.reactions[i].id for i in np.where(exclude_rxns)[0]]
        # The stoichiometric matrices should match
        stoichiometry = pd.read_csv(temp_dir / "stoichiometry.csv", index_col=0)
        true_s = create_stoichiometric_matrix(model)
        true_s = pd.DataFrame(true_s, index=[m.id for m in model.metabolites], columns=[r.id for r in model.reactions])
        pd.testing.assert_frame_equal(true_s, stoichiometry, check_names=False)
        # The dgf priors should match
        priors = pd.read_csv(temp_dir / "priors.csv", index_col=1)
        exp_dgf0_mean, exp_dgf0_cov = calc_model_dgfs_with_prediction_error(model)
        real_dgf0_mean = priors.loc[priors["parameter"] == "dgf", "loc"]
        real_priors_cov = pd.read_csv(temp_dir / "priors_cov.csv", index_col=0)
        pd.testing.assert_series_equal(exp_dgf0_mean, real_dgf0_mean, check_names=False)
        pd.testing.assert_frame_equal(exp_dgf0_cov, real_priors_cov, check_names=False)
        # The met conc measurements should match
        measurements = pd.read_csv(temp_dir / "measurements.csv", index_col=1)
        exp_log_conc_mean = pd.Series(model_struct["model"]["logConcMean"][0, 0].flatten(), index=true_s.index)
        exp_met_conc_mean = np.exp(exp_log_conc_mean)
        exp_log_conc_cov = model_struct["model"]["logConcCov"][0, 0]
        exp_log_conc_sd = pd.Series(np.sqrt(np.diag(exp_log_conc_cov)), index=true_s.index)
        real_met_conc_mean = measurements.loc[measurements["measurement_type"] == "mic", "measurement"]
        real_met_conc_sd = measurements.loc[measurements["measurement_type"] == "mic", "error_scale"]
        pd.testing.assert_series_equal(exp_met_conc_mean, real_met_conc_mean, check_names=False)
        pd.testing.assert_series_equal(exp_log_conc_sd, real_met_conc_sd, check_names=False)


@pytest.mark.usefixtures("temp_dir")
def test_gollub_files_fit_singles(temp_dir):
    """ Test that all gollub model files can be read and fitted without issues"""
    gollub_files = list((base_dir / "data" / "raw" / "from_gollub_2020").glob("**/*.mat"))
    config = load_model_configuration(str(base_dir / "tests" / "test_small_likelihood.toml"))
    assert len(gollub_files) > 0
    np.random.seed(42)
    gollub_files = np.random.choice(gollub_files, 2, replace=False)
    for file in gollub_files:
        write_gollub2020_models([file], temp_dir)
        # Run the files
        result_dir = temp_dir / "results"
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir()
        config.result_dir = result_dir
        generate_samples(config)


