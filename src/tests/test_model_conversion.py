import logging
import os
import shutil
import sys

import pytest
import multitfa
from cobra import Model, Metabolite, Reaction
from cobra.io import load_model
import pandas as pd
from multitfa.core import tmodel  # This is weird
import src.model_conversion as model_conversion
from pathlib import Path

from model_setup import ecoli_model, model_small

from src.fitting import generate_samples
from src.model_configuration import ModelConfiguration, load_model_configuration

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

@pytest.mark.usefixtures("ecoli_model")
def test_model_writing(ecoli_model):
    test_dir = Path("test_dir")
    # Check the stoichiometry of a few reactions at random
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    assert ecoli_model.reactions[0].id == "PFK"
    assert all(ecoli_model.reactions[0].cal_stoichiometric_matrix() == S["PFK"])
    assert ecoli_model.reactions[70].id == 'FUMt2_2'
    assert all(ecoli_model.reactions[70].cal_stoichiometric_matrix() == S['FUMt2_2'])
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        mean = ecoli_model.metabolites.get_by_id(id).delG_f
        sd = ecoli_model.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)

@pytest.mark.usefixtures("model_small")
def test_model_writing_small(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        mean = model_small.metabolites.get_by_id(id).delG_f
        sd = model_small.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)

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
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        mean = model_small.metabolites.get_by_id(id).delG_f
        sd = model_small.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)


