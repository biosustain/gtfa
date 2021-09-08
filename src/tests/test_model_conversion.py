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

from src.fitting import generate_samples
from src.model_configuration import ModelConfiguration, load_model_configuration

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')


def test_model_writing(test_model):
    test_dir = Path("test_dir")
    # Check the stoichiometry of a few reactions at random
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    assert test_model.reactions[0].id == "PFK"
    assert all(test_model.reactions[0].cal_stoichiometric_matrix() == S["PFK"])
    assert test_model.reactions[70].id == 'FUMt2_2'
    assert all(test_model.reactions[70].cal_stoichiometric_matrix() == S['FUMt2_2'])
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        mean = test_model.metabolites.get_by_id(id).delG_f
        sd = test_model.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)


def test_model_writing_small(test_model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Check the dgf priors
    priors = pd.read_csv(test_dir / "priors.csv")
    for rownum, row in priors[priors["parameter"] == "dgf"].iterrows():
        id = row["target_id"]
        mean = test_model_small.metabolites.get_by_id(id).delG_f
        sd = test_model_small.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)


def test_small_model_prior(test_model_small):
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
        mean = test_model_small.metabolites.get_by_id(id).delG_f
        sd = test_model_small.metabolites.get_by_id(id).std_dev
        assert row["loc"] == pytest.approx(mean)
        assert row["scale"] == pytest.approx(sd)


@pytest.fixture
def test_model():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    tmodel = build_test_model()
    test_dir = Path("test_dir")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    result_dir = test_dir / "results"
    result_dir.mkdir()
    test_dir.mkdir()
    model_conversion.write_model_files(tmodel, test_dir)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)


@pytest.fixture
def test_model_small():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    tmodel = build_small_test_model()
    test_dir = Path("test_dir")
    result_dir = test_dir / "results"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(tmodel, test_dir)
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type","target_id","condition_id","measurement","error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 1.2, 0.1]])
    header.to_csv(test_dir / "measurements.csv", index=False)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)

def build_small_test_model():
    """
    Make the small toy model with the interconversion between F6P,
    :return:
    """
    model = Model("small_toy")
    model.add_metabolites([
        Metabolite(id="f6p_c", name="Fructose 6 phosphate", compartment="c"),
        Metabolite(id="g6p_c", name="Glucose 6 phosphate", compartment="c"),
        Metabolite(id="f1p_c", name="Fructose 1 phosphate", compartment="c"),
        Metabolite(id="g1p_c", name="Glucose 1 phosphate", compartment="c")
    ])
    model.add_reactions([Reaction("g6p/g1p"),
                        Reaction("g1p/f1p"),
                        Reaction("f1p/f6p"),
                        Reaction("f6p/g6p"),
                        Reaction("f6p/g1p"),
                        ])
    # Add the reactions
    # Add the stoichiometry
    model.reactions[0].build_reaction_from_string("g6p_c <--> g1p_c"),
    model.reactions[1].build_reaction_from_string("g1p_c <--> f1p_c"),
    model.reactions[2].build_reaction_from_string("f1p_c <--> f6p_c"),
    model.reactions[3].build_reaction_from_string("f6p_c <--> g6p_c"),
    model.reactions[4].build_reaction_from_string("f6p_c <--> g1p_c")
    # Boundary reactions
    model.add_boundary(model.metabolites.get_by_id("g6p_c"), type="sink", lb=1, ub=1)
    model.add_boundary(model.metabolites.get_by_id("f1p_c"), type="exchange")
    # Make the compartment info
    compartment_info = pd.DataFrame([[7.5, 0.25, 298.15]], index=["c"], columns=["pH", "I", "T"])
    thermo_model = tmodel(model, compartment_info=compartment_info)
    # Add the KEGG ids
    thermo_model.metabolites[0].Kegg_id = "kegg:C00085"
    thermo_model.metabolites[1].Kegg_id = "kegg:C00668"
    thermo_model.metabolites[2].Kegg_id = "kegg:C05001"
    thermo_model.metabolites[3].Kegg_id = "kegg:C00103"

    return thermo_model


####### Taken directly from multitfa load_test_model
def build_test_model():
    model = load_model("e_coli_core")
    pH_I_T_dict = {
        "pH": {"c": 7.5, "e": 7, "p": 7},
        "I": {"c": 0.25, "e": 0, "p": 0},
        "T": {"c": 298.15, "e": 298.15, "p": 298.15},
    }
    del_psi_dict = {
        "c": {"c": 0, "e": 0, "p": 150},
        "e": {"c": 0, "e": 0, "p": 0},
        "p": {"c": -150, "e": 0, "p": 0},
    }
    del_psi = pd.DataFrame.from_dict(data=del_psi_dict)
    comp_info = pd.DataFrame.from_dict(data=pH_I_T_dict)
    Excl = [rxn.id for rxn in model.boundary] + [
        "BIOMASS_Ecoli_core_w_GAM",
        "O2t",
        "H2Ot",
    ]
    tfa_model = tmodel(
        model, Exclude_list=Excl, compartment_info=comp_info, membrane_potential=del_psi
    )
    for met in tfa_model.metabolites:
        kegg_id = "bigg.metabolite:" + met.id[:-2]
        met.Kegg_id = kegg_id
    tfa_model.update()
    return tfa_model
########## End of direct copy
