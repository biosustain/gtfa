"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.linalg import null_space

from .util import codify, rref, get_free_fluxes

REL_TOL = 1e-12
FUNCTION_TOL = 1e-12
MAX_NUM_STEPS = int(1e9)

@dataclass
class IndPrior1d:
    parameter_name: str
    location: pd.Series
    scale: pd.Series

@dataclass
class IndPrior2d:
    parameter_name: str
    location: pd.DataFrame
    scale: pd.DataFrame


def extract_prior_1d(
    parameter: str,
    priors: pd.DataFrame,
    coords: List[str],
    default_loc: float,
    default_scale: float
) -> IndPrior1d:
    param_priors = priors.groupby("parameter").get_group(parameter).set_index("target_id")
    loc, scale = (
        param_priors[col].reindex(coords).fillna(default)
        for col, default in [("loc", default_loc), ("scale", default_scale)]
    )
    return IndPrior1d(parameter, loc, scale)

    
def extract_prior_2d(
    parameter: str,
    priors: pd.DataFrame,
    target_coords: List[str],
    condition_coords: List[str],
    default_loc: float,
    default_scale: float
) -> IndPrior2d:
    if parameter not in priors["parameter"].unique():
        loc, scale = (
            pd.DataFrame(default, index=condition_coords, columns=target_coords)
            for default in [default_loc, default_scale]
        )
        return IndPrior2d(parameter, loc, scale)
    else:
        param_priors = priors.groupby("parameter").get_group(parameter)
        loc, scale = (
            param_priors
            .set_index(["condition_id", "target_id"])
            [col]
            .unstack()
            .reindex(condition_coords)
            .reindex(target_coords, axis=1)
            .fillna(default)
            for col, default in [("loc", default_loc), ("scale", default_scale)]
        )
        return IndPrior2d(parameter, loc, scale)


def get_coords(S: pd.DataFrame, measurements: pd.DataFrame):
    free_fluxes, free_to_fixed = get_free_fluxes(S.to_numpy())
    ff_cols = S.columns[free_fluxes]
    fixed_cols = S.columns.drop(ff_cols)
    is_transport = pd.Series(["transport" in r for r in S.columns], index=S.columns)
    transports = is_transport.loc[is_transport].index
    enzymes = is_transport.loc[~is_transport].index
    reaction_ind_map = codify(S.columns)
    # metabolite_ind_map = codify(S.index)
    # "reaction_ind": codify(S.columns)
    # "metabolite_ind": codify(S.index)
    return {
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "reaction_ind": codify(S.columns),
        "metabolite_ind": codify(S.index),
        "enzyme_names": list(enzymes),
        "enzyme_free": get_ordered_overlap(reaction_ind_map, [enzymes, ff_cols]),
        "enzyme_fixed": get_ordered_overlap(reaction_ind_map, [enzymes, fixed_cols]),
        "transport": list(transports),
        "transport_free": get_ordered_overlap(reaction_ind_map, [transports, ff_cols]),
        "transport_fixed": get_ordered_overlap(reaction_ind_map, [transports, fixed_cols]),
        "condition": list(measurements["condition_id"].unique()),
        "free_flux": list(ff_cols),
        "fixed_flux": list(fixed_cols),
        "free_to_fixed": free_to_fixed.tolist()
    }


def get_stan_input(
    measurements: pd.DataFrame,
    S: pd.DataFrame,
    priors: pd.DataFrame,
    likelihood: bool,
) -> Dict:
    """Get an input to cmdstanpy.CmdStanModel.sample.

    :param measurements: a pandas DataFrame whose rows represent measurements

    :param model_config: a dictionary with keys "priors", "likelihood" and
    "x_cols".

    """
    measurements_by_type = dict(
        measurements.groupby("measurement_type").__iter__()
    )
    for t in ["mic", "flux", "enzyme"]:
        if t not in measurements_by_type.keys():
            raise ValueError(f"No {t} measurements provided.")
    coords = get_coords(S, measurements)
    prior_dgf = extract_prior_1d("dgf", priors, coords["metabolite"], -200, 200)
    free_enzyme = get_name_ordered_overlap(coords, "reaction_ind", ["enzyme_names", "free_flux"])
    free_transport = get_name_ordered_overlap(coords, "reaction_ind", ["transport", "free_flux"])
    prior_transport_free = extract_prior_2d("transport", priors, free_transport, coords["condition"], 0.4,
                                            0.01)
    prior_b_free = extract_prior_2d("b", priors, free_enzyme, coords["condition"], 2, 2)
    prior_enzyme = extract_prior_2d("enzyme_names", priors, coords["enzyme_names"], coords["condition"], 1, 0.1)
    prior_log_metabolite = extract_prior_2d("metabolite", priors, coords["metabolite"], coords["condition"], 0, 2)
    return {
        "N_metabolite": S.shape[0],
        "N_transport": len(coords["transport"]),
        "N_enzyme": len(coords["enzyme_names"]),
        "N_reaction": S.shape[1],
        "N_free_flux": len(coords["free_flux"]),
        "N_free_enzyme": len(free_enzyme),
        "N_free_transport": len(free_transport),
        "S": S.values.tolist(),
        # Indexing
        "free_to_fixed": coords["free_to_fixed"],
        "ix_free_flux": make_index_map(coords, "reaction", ["free_flux"]),
        "ix_free_enzyme": make_index_map(coords, "reaction", ["free_flux", "enzyme_names"]),
        "ix_free_transport": make_index_map(coords, "reaction", ["free_flux", "transport"]),
        "ix_fixed_flux": make_index_map(coords, "reaction", ["fixed_flux"]),
        "ix_fixed_enzyme": make_index_map(coords, "reaction", ["fixed_flux", "enzyme_names"]),
        "ix_fixed_transport": make_index_map(coords, "reaction", ["fixed_flux", "transport"]),
        "ix_fixed_to_enzyme": make_index_map(coords, "enzyme_names", ["enzyme_names", "fixed_flux"]),
        "ix_free_to_enzyme": make_index_map(coords, "enzyme_names", ["enzyme_names", "free_flux"]),
        "ix_fixed_to_transport": make_index_map(coords, "transport", ["transport", "fixed_flux"]),
        "ix_free_to_transport": make_index_map(coords, "transport", ["transport", "free_flux"]),
        # measurements
        "N_condition": measurements["condition_id"].nunique(),
        "N_y_enzyme": len(measurements_by_type["enzyme"]),
        "N_y_metabolite": len(measurements_by_type["mic"]),
        "N_y_flux": len(measurements_by_type["flux"]),
        "y_flux": measurements_by_type["flux"]["measurement"].values.tolist(),
        "sigma_flux": measurements_by_type["flux"]["error_scale"].values.tolist(),
        "reaction_y_flux": measurements_by_type["flux"]["target_id"].map(codify(coords["reaction"])).values.tolist(),
        "condition_y_flux": measurements_by_type["flux"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "y_enzyme": measurements_by_type["enzyme"]["measurement"].values.tolist(),
        "sigma_enzyme": measurements_by_type["enzyme"]["error_scale"].values.tolist(),
        "enzyme_y_enzyme": measurements_by_type["enzyme"]["target_id"].map(codify(coords["enzyme_names"])).values.tolist(),
        "condition_y_enzyme": measurements_by_type["enzyme"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "y_metabolite": measurements_by_type["mic"]["measurement"].values.tolist(),
        "sigma_metabolite": measurements_by_type["mic"]["error_scale"].values.tolist(),
        "metabolite_y_metabolite": measurements_by_type["mic"]["target_id"].map(codify(coords["metabolite"])).values.tolist(),
        "condition_y_metabolite": measurements_by_type["mic"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        # priors
        "prior_dgf": [prior_dgf.location.values.tolist(), prior_dgf.scale.values.tolist()],
        "prior_transport_free": [prior_transport_free.location.values.tolist(), prior_transport_free.scale.values.tolist()],
        "prior_enzyme": [prior_enzyme.location.values.tolist(), prior_enzyme.scale.values.tolist()],
        "prior_b_free": [prior_b_free.location.values.tolist(), prior_b_free.scale.values.tolist()],
        "prior_log_metabolite": [prior_log_metabolite.location.values.tolist(), prior_log_metabolite.scale.values.tolist()],
        # config
        "likelihood": int(likelihood),
    }


def get_name_ordered_overlap(coords: {}, order_by: str, name_list: [str]) -> [int]:
    """
    Take a list of names of vectors and return an ordered list of indices
    :param coords: the coords dict
    :param order_by: the indexs to order by, either metabolites or reactions
    :param name_list: A list of named vectors to include
    :return:
    """
    assert order_by in ["reaction_ind", "metabolite_ind"]
    return get_ordered_overlap(coords[order_by], [coords[name] for name in name_list])


def get_ordered_overlap(order_by: {}, inds_lists: [list]) -> []:
    """
    :param order_by: A dict that defines the order of the indices
    :param inds: A list of lists of indices
    :return:
    """
    # Get the unique indices of the overlap
    unique_inds = set(inds_lists[0]).intersection(*inds_lists)
    sorted_inds = sorted(unique_inds, key=order_by.get)
    assert not any([ind is None for ind in sorted_inds]), "All indices should be in the sorting dict"
    return sorted_inds


def make_index_map(coords: dict, to_name: str, from_names: [str]):
    """
    Make a map from a list of lists of names defining overlapping conditions
    :param coords: The coords dict
    :param to_name: The name of the vector that should be mapped to
    :param from_names: A list of names whose union should be mapped from
    :return: A list of 1-based stan indices
    """
    codified = codify(coords[to_name])  # Indices in stan 1-based indexing
    ordered_overlap = get_ordered_overlap(codified, [coords[name] for name in from_names])
    # Needs to loop through the entire first vector to maintain ordering
    ind_map = [codified.get(ind, 0) for ind in ordered_overlap]
    return ind_map
