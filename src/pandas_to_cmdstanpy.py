"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.linalg import null_space

from .util import codify, rref, get_free_fluxes

logger = logging.getLogger(__name__)

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
    # List of prefixes for non-internal reactions. Transport, Exchange, Sink, Demand, Excluded
    # Make sure they are protected for the regular expression
    TRANSPORT_NAMES = [re.escape(name) for name in ["transport", "EX_", "SK_", "DM_", "EXCL_"]]
    # Search for any of the names using regex
    is_transport = S.columns.str.contains("|".join(TRANSPORT_NAMES))
    transports = S.columns[is_transport]
    enzymes = S.columns[~is_transport]
    num_met, num_rxn = S.shape
    num_trans = is_transport.sum()
    # Calculate the final matrix and the free variables
    s_gamma = S.T[~is_transport]
    s_gamma_mod = np.zeros((num_rxn, num_met + num_trans))
    s_gamma_mod[:num_trans, :num_trans] = np.identity(num_trans)
    s_gamma_mod[num_trans:, num_trans:] = s_gamma.to_numpy()
    s_total = S @ s_gamma_mod
    free_x_ind, _ = get_free_fluxes(s_total.to_numpy())
    # Get the fixed and free x values
    x_names = pd.Series(transports.tolist() + S.index.tolist())
    free_x_names = x_names[free_x_ind]
    fixed_x_names = x_names[~free_x_ind]
    # A list of indices for each free x
    free_x = np.arange(1, len(free_x_ind)+1)[free_x_ind]
    fixed_x = np.arange(1, len(free_x_ind)+1)[~free_x_ind]
    # This is a vector with transport reactions first followed by the metabolites
    transport_free_ind = free_x_ind[:num_trans]
    transport_free = transports[transport_free_ind]
    transport_fixed = transports[~transport_free_ind]
    conc_free_ind = free_x_ind[num_trans:]
    conc_free = S.index[conc_free_ind]
    conc_fixed = S.index[~conc_free_ind]
    reaction_ind_map = codify(S.columns)
    met_ind_map = codify(S.index)
    return {
        # Maps to stan indices
        "reaction_ind": reaction_ind_map,
        "metabolite_ind": met_ind_map,
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "x_names": list(x_names),
        "free_x_names": list(free_x_names),
        "fixed_x_names": list(fixed_x_names),
        "enzyme_names": list(enzymes),
        "transport": list(transports),
        "free_transport": list(transport_free),
        "fixed_transport": list(transport_fixed),
        "free_met_conc": list(conc_free),
        "fixed_met_conc": list(conc_fixed),
        "condition": list(measurements["condition_id"].unique()),
        "free_x": list(free_x),
        "fixed_x": list(fixed_x),
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
    if len(measurements) == 0:
        raise ValueError("At least one measurement is required")
    # Make a dictionary based on the observation type
    measurements_by_type = dict(measurements.groupby("measurement_type").__iter__())
    for t in ["mic", "flux", "enzyme"]:
        if t not in measurements_by_type:
            logger.warning(f"No {t} measurements provided.")
            measurements_by_type[t] = pd.DataFrame(columns=measurements.columns)
    coords = get_coords(S, measurements)
    free_transport = get_name_ordered_overlap(coords, "reaction_ind", ["transport", "free_x_names"])
    free_met_conc = get_name_ordered_overlap(coords, "metabolite_ind", ["metabolite", "free_x_names"])
    prior_b = extract_prior_2d("b", priors, coords["enzyme_names"], coords["condition"], 2, 2)
    prior_enzyme = extract_prior_2d("enzyme_names", priors, coords["enzyme_names"], coords["condition"], 1, 0.1)
    prior_met_conc_free = extract_prior_2d("metabolite", priors, free_met_conc, coords["condition"], 0, 2)
    prior_transport_free = extract_prior_2d("transport", priors, free_transport, coords["condition"], 0.4, 0.01)
    prior_dgf = extract_prior_1d("dgf", priors, coords["metabolite"], -200, 200)
    return {
        # Sizes
        "N_metabolite": S.shape[0],
        "N_reaction": S.shape[1],
        "N_transport": len(coords["transport"]),
        "N_enzyme": len(coords["enzyme_names"]),
        "N_fixed_transport": len(coords["fixed_transport"]),
        "N_free_transport": len(coords["free_transport"]),
        "N_fixed_met_conc": len(coords["fixed_met_conc"]),
        "N_free_met_conc": len(coords["free_met_conc"]),
        "N_free_x": len(coords["free_x"]),
        "N_fixed_x": len(coords["fixed_x"]),
        "N_x": len(coords["fixed_x"] + coords["fixed_x"]),
        # Network
        "S": S.values.tolist(),
        # Indexing
        "ix_free_to_met": make_index_map(coords, "free_x_names", ["free_met_conc"]),
        "ix_fixed_to_met": make_index_map(coords, "fixed_x_names", ["fixed_met_conc"]),
        "ix_free_to_trans": make_index_map(coords, "transport", ["free_transport"]),
        "ix_fixed_to_trans": make_index_map(coords, "transport", ["fixed_transport"]),
        "ix_enzyme": make_index_map(coords, "reaction", ["enzyme_names"]),
        "ix_transport": make_index_map(coords, "reaction", ["transport"]),
        "ix_free": make_index_map(coords, "x_names", ["free_x_names"]),
        "ix_fixed": make_index_map(coords, "x_names", ["fixed_x_names"]),
        "ix_free_transport": make_index_map(coords, "reaction", ["transport", "free_x_names"]),
        "ix_fixed_transport": make_index_map(coords, "reaction", ["transport", "fixed_x_names"]),
        "ix_free_met_conc": make_index_map(coords, "metabolite", ["metabolite", "free_x_names"]),
        "ix_fixed_met_conc": make_index_map(coords, "metabolite", ["metabolite", "fixed_x_names"]),
        # measurements
        "N_condition": pd.concat([measurements["condition_id"], priors["condition_id"]]).nunique(),
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
        "prior_b": [prior_b.location.values.tolist(), prior_b.scale.values.tolist()],
        "prior_free_met_conc": [prior_met_conc_free.location.values.tolist(), prior_met_conc_free.scale.values.tolist()],
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
