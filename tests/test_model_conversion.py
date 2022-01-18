import logging
from pathlib import Path

from cobra.util.array import create_stoichiometric_matrix
import cobra.util.array
import numpy as np
import pandas as pd
import pytest

from src.fitting import generate_samples
from src.model_configuration import load_model_configuration
# Don't delete
# from model_setup import ecoli_model, model_small
from model_setup import ecoli_model, model_small
from src.model_conversion import calc_model_dgfs

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')


@pytest.mark.usefixtures("ecoli_model")
def test_model_writing(ecoli_model):
    test_dir = Path("test_dir")
    # Check the stoichiometry of a few reactions at random
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    model_s = create_stoichiometric_matrix(ecoli_model)
    assert ecoli_model.reactions[0].id == "PFK"
    assert all(model_s[:, 0] == S["PFK"])
    assert ecoli_model.reactions[70].id == 'FUMt2_2'
    assert all(model_s[:, 70] == S['FUMt2_2'])
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs(ecoli_model)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(test_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())


@pytest.mark.usefixtures("model_small")
def test_model_writing_small(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs(model_small)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(test_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())


@pytest.mark.usefixtures("model_small")
def test_small_model_prior(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    result_dir = test_dir / "results"
    # Make a config and run this test model
    config = load_model_configuration("test_small_prior.toml")
    config.result_dir = result_dir
    # Run the sampling
    generate_samples(config)
    # Check results files
    priors = pd.read_csv(test_dir / "priors.csv")
    calc_dgf_mean, calc_dgf_cov = calc_model_dgfs(model_small)
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        assert row["loc"] == pytest.approx(calc_dgf_mean[id])
    # Test the covariance matrix
    file_cov = pd.read_csv(test_dir / "priors_cov.csv", index_col=0)
    np.testing.assert_array_almost_equal(calc_dgf_cov, file_cov.to_numpy())
