import logging
import shutil
import sys
from pathlib import Path

from equilibrator_api import ComponentContribution
import pandas as pd
import pytest
from cobra import Model, Metabolite, Reaction
from cobra.io import load_model

from src import model_conversion

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# This makes the tests run faster
cc = None


@pytest.fixture
def ecoli_model():
    global cc
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    tmodel = load_model("e_coli_core")
    test_dir = Path("test_dir")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    result_dir = test_dir / "results"
    result_dir.mkdir(parents=True)
    if cc is None:
        cc = ComponentContribution()
    model_conversion.write_model_files(tmodel, test_dir, cc=cc)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)


@pytest.fixture
def model_small():
    global cc
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
    if cc is None:
        cc = ComponentContribution()
    model_conversion.write_model_files(tmodel, test_dir, cc=cc)
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type","target_id","condition_id","measurement","error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(test_dir / "measurements.csv", index=False)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)


@ pytest.fixture
def model_small_rankdef():
    global cc
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    tmodel = build_small_test_model_rankdef_stoich()
    test_dir = Path("test_dir")
    result_dir = test_dir / "results"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    result_dir.mkdir()
    if cc is None:
        cc = ComponentContribution()
    model_conversion.write_model_files(tmodel, test_dir, cc=cc)
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type","target_id","condition_id","measurement","error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(test_dir / "measurements.csv", index=False)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)

@pytest.fixture
def model_small_rankdef_thermo():
    global cc
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    tmodel = build_small_test_model_rankdef_thermo()
    test_dir = Path("test_dir")
    result_dir = test_dir / "results"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    result_dir.mkdir()
    if cc is None:
        cc = ComponentContribution()
    model_conversion.write_model_files(tmodel, test_dir, cc=cc)
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type","target_id","condition_id","measurement","error_scale"],
                          data=[["mic", "f6p_c", "condition_1", 0.001, 0.1]])
    header.to_csv(test_dir / "measurements.csv", index=False)
    yield tmodel
    # Clean up
    shutil.rmtree(test_dir)



def build_small_test_model_rankdef_stoich():
    """ From the example from Metabolic control theory: A structural approach
    https://doi.org/10.1016/S0022-5193(88)80073-0

    The exact metabolites are unimportant and the row rank deficiency is important
    """
    model = Model("small_toy")
    model.add_metabolites([
        Metabolite(id="f6p_c", name="Fructose 6 phosphate", compartment="c"),
        Metabolite(id="g6p_c", name="Glucose 6 phosphate", compartment="c"),
        Metabolite(id="ATP", name="ATP", compartment="c"),
        Metabolite(id="f1p_c", name="Fructose 1 phosphate", compartment="c"),
        Metabolite(id="ADP", name="ADP", compartment="c"),
        Metabolite(id="g1p_c", name="Glucose 1 phosphate", compartment="c")

    ])
    model.add_reactions([Reaction("1"),
                         Reaction("2")
                         ])
    # Add the reactions
    # Add the stoichiometry
    model.reactions[0].build_reaction_from_string("f6p_c + ADP <--> f1p_c + ATP")
    model.reactions[1].build_reaction_from_string("g6p_c + ATP <--> g1p_c + ADP")
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
        Metabolite(id="ATP", name="ATP", compartment="c"),
        Metabolite(id="ADP", name="ADP", compartment="c"),
        Metabolite(id="pi", name="Orhtophosphate", compartment="c"),
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
    model.reactions[2].build_reaction_from_string("fum_c + ATP <--> 2obut_c + ADP + pi")
    model.reactions[3].build_reaction_from_string("acac_c + ATP <--> male_c + ADP + pi")
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
    model.metabolites[6 ].annotation = {"kegg.compound": "C00009"}
    return model
