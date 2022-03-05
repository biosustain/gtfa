import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from equilibrator_api import ComponentContribution
import pandas as pd
import pytest
from cobra import Model, Metabolite, Reaction
from cobra.io import load_model

from src import model_conversion, util
from src.dgf_estimation import calc_model_dgfs_with_prediction_error
from src.pandas_to_cmdstanpy import DEFAULT_MET_CONC_MEAN, DEFAULT_MET_CONC_SCALE
from src.util import get_smat_df

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# This makes the tests run faster
base_dir = Path(__file__).parent.parent
cc = ComponentContribution()



@pytest.fixture
def temp_dir():
    temp_dir = Path(base_dir / "tests" / "temp_dir").absolute()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def ecoli_model():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = load_model("e_coli_core")
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = calc_model_dgfs_with_prediction_error(model, cc)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    result_dir = temp_dir / "results"
    result_dir.mkdir(parents=True)
    # Get the dgfs
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = calc_model_dgfs_with_prediction_error(model, cc)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small_rankdef():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model_rankdef_stoich()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = model_conversion.calc_model_dgfs_with_prediction_error(model, cc)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small_rankdef_thermo():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model_rankdef_thermo()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = model_conversion.calc_model_dgfs_with_prediction_error(model, cc)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


def build_small_test_model_rankdef_stoich():
    """ From the example from Metabolic control theory: A structural approach
    https://doi.org/10.1016/S0022-5193(88)80073-0

    The exact metabolites are unimportant and the row rank deficiency is important
    """
    model = Model("small_toy")
    model.add_metabolites([
        Metabolite(id="f6p_c", name="Fructose 6 phosphate", compartment="c"),
        Metabolite(id="g6p_c", name="Glucose 6 phosphate", compartment="c"),
        Metabolite(id="atp_c", name="ATP", compartment="c"),
        Metabolite(id="f1p_c", name="Fructose 1 phosphate", compartment="c"),
        Metabolite(id="adp_c", name="ADP", compartment="c"),
        Metabolite(id="g1p_c", name="Glucose 1 phosphate", compartment="c")

    ])
    model.add_reactions([Reaction("1"),
                         Reaction("2")
                         ])
    # Add the reactions
    # Add the stoichiometry
    model.reactions[0].build_reaction_from_string("f6p_c + adp_c <--> f1p_c + atp_c")
    model.reactions[1].build_reaction_from_string("g6p_c + atp_c <--> g1p_c + adp_c")
    # Boundary reactions
    model.add_boundary(model.metabolites.get_by_id("f6p_c"), type="sink")
    model.add_boundary(model.metabolites.get_by_id("f1p_c"), type="exchange")
    model.add_boundary(model.metabolites.get_by_id("g6p_c"), type="exchange")
    model.add_boundary(model.metabolites.get_by_id("g1p_c"), type="exchange")
    # Make the compartment info
    # Add the KEGG ids
    model.metabolites[0].annotation = {"kegg.compound": "C00085"}
    model.metabolites[1].annotation = {"kegg.compound": "C00668"}
    model.metabolites[2].annotation = {"kegg.compound": "C05001"}
    model.metabolites[3].annotation = {"kegg.compound": "C00103"}
    model.metabolites[4].annotation = {"kegg.compound": "C00002"}
    model.metabolites[5].annotation = {"kegg.compound": "C00008"}
    model.compartment_conditions = make_default_compartment_conditions(model, cc)
    return model


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
    # Add the KEGG ids
    model.metabolites[0].annotation = {"kegg.compound": "C00085"}
    model.metabolites[1].annotation = {"kegg.compound": "C00668"}
    model.metabolites[2].annotation = {"kegg.compound": "C05001"}
    model.metabolites[3].annotation = {"kegg.compound": "C00103"}
    model.compartment_conditions = make_default_compartment_conditions(model, cc)
    return model


# This test case doesn't produce a reduced rank dgf covaraince matrix
def build_small_test_model_rankdef_thermo():
    """ From the example from Metabolic control theory: A structural approach
    https://doi.org/10.1016/S0022-5193(88)80073-0

    The exact metabolites are unimportant and the row rank deficiency is important
    """
    model = Model("small_toy")
    model.add_metabolites([
        Metabolite(id="fum_c", name="Fumarate", compartment="c"),
        Metabolite(id="male_c", name="Maleate", compartment="c"),
        Metabolite(id="2obut_c", name="2 Oxobutyrate", compartment="c"),
        Metabolite(id="acac_c", name="ACetoacetate", compartment="c"),
        Metabolite(id="atp_c", name="ATP", compartment="c"),
        Metabolite(id="adp_c", name="ADP", compartment="c"),
        Metabolite(id="pi_c", name="Orhtophosphate", compartment="c"),
    ])
    model.add_reactions([Reaction("1"),
                         Reaction("2"),
                         Reaction("3"),
                         Reaction("4")
                         ])
    # Add the reactions
    # Add the stoichiometry
    model.reactions[0].build_reaction_from_string("fum_c <--> male_c")
    model.reactions[1].build_reaction_from_string("2obut_c <--> acac_c")
    model.reactions[2].build_reaction_from_string("fum_c + atp_c <--> 2obut_c + adp_c + pi_c")
    model.reactions[3].build_reaction_from_string("acac_c + atp_c <--> male_c + adp_c + pi_c")
    # Boundary reactions
    model.add_boundary(model.metabolites.get_by_id("fum_c"), type="sink")
    model.add_boundary(model.metabolites.get_by_id("male_c"), type="exchange")
    # Make the compartment info
    # Add the KEGG ids
    model.metabolites[0].annotation = {"kegg.compound": "C00122"}
    model.metabolites[1].annotation = {"kegg.compound": "C01384"}
    model.metabolites[2].annotation = {"kegg.compound": "C00109"}
    model.metabolites[3].annotation = {"kegg.compound": "C00164"}
    model.metabolites[4].annotation = {"kegg.compound": "C00002"}
    model.metabolites[5].annotation = {"kegg.compound": "C00008"}
    model.metabolites[6].annotation = {"kegg.compound": "C00009"}
    model.compartment_conditions = make_default_compartment_conditions(model, cc)
    return model


def build_small_test_model_exchanges():
    """ From the example from Metabolic control theory: A structural approach
    https://doi.org/10.1016/S0022-5193(88)80073-0

    The exact metabolites are unimportant and the row rank deficiency is important
    """
    model = Model("small_toy")
    model.add_metabolites([
        Metabolite(id="fum_c", name="Fumarate", compartment="c"),
        Metabolite(id="male_c", name="Maleate", compartment="c"),
        Metabolite(id="2obut_c", name="2 Oxobutyrate", compartment="c"),
        Metabolite(id="acac_c", name="ACetoacetate", compartment="c"),
        Metabolite(id="atp_c", name="ATP", compartment="c"),
        Metabolite(id="adp_c", name="ADP", compartment="c"),
        Metabolite(id="pi_c", name="Orhtophosphate", compartment="c"),

        Metabolite(id="fum_p", name="Fumarate", compartment="p"),
        Metabolite(id="male_p", name="Maleate", compartment="p"),
        Metabolite(id="2obut_p", name="2 Oxobutyrate", compartment="p"),
        Metabolite(id="acac_p", name="ACetoacetate", compartment="p"),
        Metabolite(id="atp_p", name="ATP", compartment="p"),
        Metabolite(id="adp_p", name="ADP", compartment="p"),
        Metabolite(id="pi_p", name="Orhtophosphate", compartment="p"),
    ])
    model.add_reactions([Reaction("1"),
                         Reaction("2"),
                         Reaction("3"),
                         Reaction("4"),
                         Reaction("5"),
                         Reaction("6"),
                         Reaction("7"),
                         Reaction("8"),
                         Reaction("9"),
                         Reaction("10")
                         ])
    # Add the reactions
    # Add the stoichiometry
    model.reactions[0].build_reaction_from_string("fum_c <--> male_c")
    model.reactions[1].build_reaction_from_string("2obut_c <--> acac_c")
    model.reactions[2].build_reaction_from_string("fum_c + atp_c <--> 2obut_c + adp_c + pi_c")
    model.reactions[3].build_reaction_from_string("acac_c + atp_c <--> male_c + adp_c + pi_c")

    model.reactions[4].build_reaction_from_string("fum_c <--> fum_p")
    model.reactions[5].build_reaction_from_string("male_c <--> male_p")
    model.reactions[6].build_reaction_from_string("2obut_c <--> 2obut_p")
    model.reactions[7].build_reaction_from_string("acac_c <--> acac_p")
    model.reactions[8].build_reaction_from_string("acac_c <--> acac_p")
    model.reactions[9].build_reaction_from_string("pi_c <--> pi_p")
    # Boundary reactions
    model.add_boundary(model.metabolites.get_by_id("fum_p"), type="sink")
    model.add_boundary(model.metabolites.get_by_id("male_p"), type="exchange")
    # Make the compartment info
    # Add the KEGG ids
    model.metabolites[0].annotation = {"kegg.compound": "C00122"}
    model.metabolites[1].annotation = {"kegg.compound": "C01384"}
    model.metabolites[2].annotation = {"kegg.compound": "C00109"}
    model.metabolites[3].annotation = {"kegg.compound": "C00164"}
    model.metabolites[4].annotation = {"kegg.compound": "C00002"}
    model.metabolites[5].annotation = {"kegg.compound": "C00008"}
    model.metabolites[6].annotation = {"kegg.compound": "C00009"}


    model.metabolites[7].annotation = {"kegg.compound": "C00122"}
    model.metabolites[8].annotation = {"kegg.compound": "C01384"}
    model.metabolites[9].annotation = {"kegg.compound": "C00109"}
    model.metabolites[10].annotation = {"kegg.compound": "C00164"}
    model.metabolites[11].annotation = {"kegg.compound": "C00002"}
    model.metabolites[12].annotation = {"kegg.compound": "C00008"}
    model.metabolites[13].annotation = {"kegg.compound": "C00009"}
    model.compartment_conditions = pd.DataFrame([[8.0, 0.5, 298.15, 5.0],
                                                 [6.5, 0.1, 298.15, 0.2]],
                                                index=["c", "p"], columns=["pH", "I", "T", "p_mg"])
    return model


def make_default_compartment_conditions(model, cc: ComponentContribution):
    compartment_conditions = pd.DataFrame(index=model.compartments, columns=["pH", "I", "T", "p_mg"])
    for compartment in model.compartments.keys():
        compartment_conditions.loc[compartment] = [cc.p_h.m_as(""), cc.ionic_strength.m_as("M"), cc.temperature.m_as("K"), cc.p_mg.m_as("")]
    return compartment_conditions


def gen_random_log_concs(model):
    np.random.seed(0)
    n_mets = len(model.metabolites)
    locs = np.full(n_mets, DEFAULT_MET_CONC_MEAN)
    scales = np.full(n_mets, DEFAULT_MET_CONC_SCALE)
    return np.random.normal(locs, scales), np.full(n_mets, DEFAULT_MET_CONC_SCALE)
