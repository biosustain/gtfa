"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""
import itertools
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

# TAKEN FROM PTA (I think they fitted a lognormal to all met concentrations in all conditions)
DEFAULT_MET_CONC_MEAN = -8.3371
DEFAULT_MET_CONC_SCALE = 1.9885
# This still needs to be determined
DEFAULT_ENZ_CONC_MEAN = -8.3371
DEFAULT_ENZ_CONC_SCALE = 1.9885
DEFAULT_EXCHANGE_MEAN = 0
DEFAULT_EXCHANGE_SCALE = 1
DEFAULT_B_MEAN = 3
DEFAULT_B_SCALE = 3


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

    def to_dataframe(self, measurement_type):
        location = self.location.unstack()
        scale = self.scale.unstack()
        df = pd.concat([location, scale], axis=1).reset_index()
        df.columns = ["parameter", "condition_id", "loc", "scale"]
        df["measurement_type"] = measurement_type
        return df

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


def get_exchange_rxns(S):
    # List of prefixes for non-internal reactions. transport, Exchange, Sink, Demand, Excluded
    EXCHANGE_NAMES = [re.escape(name) for name in ["transport", "exchange", "EX_", "SK_", "DM_", "EXCL_"]]
    # Search for any of the names using regex
    return S.columns.str.contains("|".join(EXCHANGE_NAMES))


def get_coords(S: pd.DataFrame, measurements: pd.DataFrame, priors: pd.DataFrame, order=None):
    if not type(priors) == pd.DataFrame:
        raise RuntimeError("priors must be a dataframe")
    # Make sure they are protected for the regular expression
    is_exchange = get_exchange_rxns(S)
    base_ordering = S.columns[is_exchange].to_series().append(S.index.to_series())
    exchanges = S.columns[is_exchange]
    internals = S.columns[~is_exchange]
    num_ex = is_exchange.sum()
    free_x_ind = get_free_x(S, order)
    # Get the fixed and free x values
    x_names = base_ordering
    free_x_names = x_names[free_x_ind]
    fixed_x_names = x_names[~free_x_ind]
    # A list of indices for each free x
    free_x = np.arange(1, len(free_x_ind)+1)[free_x_ind]
    fixed_x = np.arange(1, len(free_x_ind)+1)[~free_x_ind]
    # This is a vector with exchange reactions first followed by the metabolites
    exchange_free_ind = free_x_ind[:num_ex]
    exchange_free = exchanges[exchange_free_ind]
    exchange_fixed = exchanges[~exchange_free_ind]
    conc_free_ind = free_x_ind[num_ex:]
    conc_free = S.index[conc_free_ind]
    conc_fixed = S.index[~conc_free_ind]
    reaction_ind_map = codify(S.columns)
    met_ind_map = codify(S.index)
    # Get a list of all conditions
    measurement_conditions = measurements["condition_id"]
    prior_conditions = priors["condition_id"][~priors["condition_id"].isna()]
    conditions = measurement_conditions.append(prior_conditions).unique()
    return {
        # Maps to stan indices
        "reaction_ind": reaction_ind_map,
        "metabolite_ind": met_ind_map,
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "x_names": list(x_names),
        "free_x_names": list(free_x_names),
        "fixed_x_names": list(fixed_x_names),
        "internal_names": list(internals),
        "exchange": list(exchanges),
        "free_exchange": list(exchange_free),
        "fixed_exchange": list(exchange_fixed),
        "free_met_conc": list(conc_free),
        "fixed_met_conc": list(conc_fixed),
        "condition": list(conditions),
        "free_x": list(free_x),
        "fixed_x": list(fixed_x),
    }


def get_free_x(S, order):
    """
    Get the free x values of the corresponding S_c matrix.
    :param S: The stoichiometric matrix
    :param order: A list of x values in the desired order from free to fixed. If the list is incomplete then the rest of the
    x values will be filled.
    :return:
    """
    # Determine the exchange reactions
    is_exchange = get_exchange_rxns(S)
    exchange_names = S.columns[is_exchange]
    base_ordering = S.columns[is_exchange].to_series().append(S.index.to_series())
    met_names = base_ordering[~base_ordering.isin(exchange_names)]
    # Convert an incomplete ordering into a full ordering
    if order is None:
        order = base_ordering
    else:
        # Push the ordered columns to the front and fill in the rest
        order = pd.Series(order, index=order)
        order = order.append(base_ordering.drop(order))
        # Now reverse the order to put the "freest" variables on the right
        order = order.iloc[::-1]
    num_ex = is_exchange.sum()
    # Calculate the final matrix and the free variables
    s_x = pd.DataFrame(0, columns=base_ordering, index=S.columns)
    s_x.loc[exchange_names, exchange_names] = np.identity(num_ex)
    s_x.loc[~is_exchange, met_names] = S.T[~is_exchange].to_numpy()
    s_total = S @ s_x
    # Reorder the columns according the the given ordering
    s_total = s_total.loc[:, order]
    free_x_ind, _ = get_free_fluxes(s_total.to_numpy())
    # Revert back to original ordering
    free_x_ind = free_x_ind[order.index.get_indexer(base_ordering)]
    return pd.Series(free_x_ind, index=base_ordering)


def check_input(measurements, priors):
    if len(measurements) == 0:
        raise ValueError("At least one measurement is required")
    measurements_by_type = dict(measurements.groupby("measurement_type").__iter__())
    # Check that enzyme and metabolite measurements are in log scale
    if "enzyme" in measurements_by_type and not measurements_by_type["enzyme"]["measurement"].between(0, 1).all():
        raise ValueError("Enzyme concentration measurements should be between 0 and 1 molar"
                         "Are they maybe recorded as log concentrations?")
    if "mic" in measurements_by_type and not measurements_by_type["mic"]["measurement"].between(0, 1).all():
        raise ValueError("Metabolite concentration measurements should be between 0 and 1 molar. "
                         "Are they maybe recorded as log concentrations?")
    priors_by_type = dict(priors.groupby("parameter").__iter__())
    # Check that the lognormal priors are in the correct range
    if ("enzyme" in priors_by_type and not priors_by_type["enzyme"]["loc"].between(-20, 0).all()) \
            or ("concentration" in priors_by_type and not priors_by_type["concentration"]["loc"].between(-20, 0).all()):
        raise ValueError("Reasonable lognormal concentration priors should be between -20 and 0. "
                         "Maybe you made a mistake in the formulation?")


def get_stan_input(
    measurements: pd.DataFrame,
    S: pd.DataFrame,
    priors: pd.DataFrame,
    priors_cov: pd.DataFrame,
    likelihood: bool,
    order=None) -> Dict:
    """Get an input to cmdstanpy.CmdStanModel.sample.

    :param measurements: a pandas DataFrame whose rows represent measurements

    :param model_config: a dictionary with keys "priors", "likelihood" and
    "x_cols".

    """
    check_input(measurements, priors)
    coords = get_coords(S, measurements, priors, order)
    measurements_by_type = group_measurement_types(likelihood, measurements)
    free_exchange = get_name_ordered_overlap(coords, "reaction_ind", ["exchange", "free_x_names"])
    free_met_conc = get_name_ordered_overlap(coords, "metabolite_ind", ["metabolite", "free_x_names"])
    prior_b = extract_prior_2d("b", priors, coords["internal_names"], coords["condition"], DEFAULT_B_MEAN, DEFAULT_B_SCALE)
    prior_enzyme = extract_prior_2d("internal_names", priors, coords["internal_names"], coords["condition"], DEFAULT_ENZ_CONC_MEAN, DEFAULT_ENZ_CONC_SCALE)
    prior_met_conc_free = extract_prior_2d("metabolite", priors, free_met_conc, coords["condition"], DEFAULT_MET_CONC_MEAN, DEFAULT_MET_CONC_SCALE)
    prior_exchange_free = extract_prior_2d("exchange", priors, free_exchange, coords["condition"], DEFAULT_EXCHANGE_MEAN, DEFAULT_EXCHANGE_SCALE)
    # Add the fixed priors to the measurements
    fixed_exchange_prior_df, fixed_met_prior_df = fixed_prior_to_measurements(coords, priors)
    measurements_by_type["mic"] = measurements_by_type["mic"].append(fixed_met_prior_df)
    measurements_by_type["flux"] = measurements_by_type["flux"].append(fixed_exchange_prior_df)
    # We're going to assume full prior information on dgf
    prior_dgf_mean = priors[priors["parameter"] == "dgf"]["loc"]
    if len(prior_dgf_mean) != S.shape[0]:
        raise ValueError("All dgf means must be provided in the priors file")
    return {
        # Sizes
        "N_metabolite": S.shape[0],
        "N_reaction": S.shape[1],
        "N_exchange": len(coords["exchange"]),
        "N_internal": len(coords["internal_names"]),
        "N_fixed_exchange": len(coords["fixed_exchange"]),
        "N_free_exchange": len(coords["free_exchange"]),
        "N_fixed_met_conc": len(coords["fixed_met_conc"]),
        "N_free_met_conc": len(coords["free_met_conc"]),
        "N_free_x": len(coords["free_x"]),
        "N_fixed_x": len(coords["fixed_x"]),
        "N_x": len(coords["fixed_x"] + coords["free_x"]),
        # Network
        "S": S.values.tolist(),
        # Indexing
        "ix_free_met_to_free": make_index_map(coords, "free_x_names", ["free_met_conc"]),
        "ix_free_ex_to_free": make_index_map(coords, "free_x_names", ["free_exchange"]),
        "ix_fixed_met_to_fixed": make_index_map(coords, "fixed_x_names", ["fixed_met_conc"]),
        "ix_fixed_ex_to_fixed": make_index_map(coords, "fixed_x_names", ["fixed_exchange"]),
        "ix_free_to_x": make_index_map(coords, "x_names", ["free_x_names"]),
        "ix_fixed_to_x": make_index_map(coords, "x_names", ["fixed_x_names"]),
        "ix_ex_to_x": make_index_map(coords, "x_names", ["exchange"]),
        "ix_met_to_x": make_index_map(coords, "x_names", ["metabolite"]),
        "ix_internal_to_rxn": make_index_map(coords, "reaction", ["internal_names"]),
        "ix_ex_to_rxn": make_index_map(coords, "reaction", ["exchange"]),
        "ix_free_met_to_met": make_index_map(coords, "metabolite", ["free_met_conc"]),
        "ix_fixed_met_to_met": make_index_map(coords, "metabolite", ["fixed_met_conc"]),
        "ix_free_ex_to_ex": make_index_map(coords, "exchange", ["exchange", "free_x_names"]),
        "ix_fixed_ex_to_ex": make_index_map(coords, "exchange", ["exchange", "fixed_x_names"]),
        # measurements
        "N_condition": len(coords["condition"]),
        "N_y_enzyme": len(measurements_by_type["enzyme"]),
        "N_y_metabolite": len(measurements_by_type["mic"]),
        "N_y_flux": len(measurements_by_type["flux"]),
        "y_flux": measurements_by_type["flux"]["measurement"].values.tolist(),
        "sigma_flux": measurements_by_type["flux"]["error_scale"].values.tolist(),
        "reaction_y_flux": measurements_by_type["flux"]["target_id"].map(codify(coords["reaction"])).values.tolist(),
        "condition_y_flux": measurements_by_type["flux"]["condition_id"].map(
            codify(coords["condition"])).values.tolist(),
        # Concentrations given on a log scale
        "y_enzyme": np.log(measurements_by_type["enzyme"]["measurement"]).values.tolist(),
        "sigma_enzyme": measurements_by_type["enzyme"]["error_scale"].values.tolist(),
        "internal_y_enzyme": measurements_by_type["enzyme"]["target_id"].map(
            codify(coords["internal_names"])).values.tolist(),
        "condition_y_enzyme": measurements_by_type["enzyme"]["condition_id"].map(
            codify(coords["condition"])).values.tolist(),
        # Concentrations given on a log scale
        "y_metabolite": np.log(measurements_by_type["mic"]["measurement"]).values.tolist(),
        "sigma_metabolite": measurements_by_type["mic"]["error_scale"].values.tolist(),
        "metabolite_y_metabolite": measurements_by_type["mic"]["target_id"].map(
            codify(coords["metabolite"])).values.tolist(),
        "condition_y_metabolite": measurements_by_type["mic"]["condition_id"].map(
            codify(coords["condition"])).values.tolist(),
        # priors
        "prior_dgf_mean": prior_dgf_mean.values.tolist(),
        "prior_dgf_cov": priors_cov.values.tolist(),
        "prior_exchange_free": [prior_exchange_free.location.values.tolist(),
                                prior_exchange_free.scale.values.tolist()],
        "prior_enzyme": [prior_enzyme.location.values.tolist(), prior_enzyme.scale.values.tolist()],
        "prior_b": [prior_b.location.values.tolist(), prior_b.scale.values.tolist()],
        "prior_free_met_conc": [prior_met_conc_free.location.values.tolist(),
                                prior_met_conc_free.scale.values.tolist()],
    }


def group_measurement_types(likelihood, measurements):
    # If likelihood is off, then remove the measurements
    if not likelihood:
        measurements = pd.DataFrame(columns=measurements.columns)
    # Make a dictionary based on the observation type
    measurements_by_type = dict(measurements.groupby("measurement_type").__iter__())
    for t in ["mic", "flux", "enzyme"]:
        if t not in measurements_by_type:
            # Only warn if likelihood is on
            if likelihood:
                logger.warning(f"No {t} measurements provided.")
            measurements_by_type[t] = pd.DataFrame(columns=measurements.columns)
    return measurements_by_type


def fixed_prior_to_measurements(coords, priors):
    """
    Convert the fixed exchange and met conc priors to measurements.
    """
    fixed_exchange = get_name_ordered_overlap(coords, "reaction_ind", ["exchange", "fixed_x_names"])
    fixed_met_conc = get_name_ordered_overlap(coords, "metabolite_ind", ["metabolite", "fixed_x_names"])
    prior_met_conc_fixed = extract_prior_2d("metabolite", priors, fixed_met_conc, coords["condition"],
                                            DEFAULT_MET_CONC_MEAN, DEFAULT_MET_CONC_SCALE)
    prior_exchange_fixed = extract_prior_2d("exchange", priors, fixed_exchange, coords["condition"],
                                            DEFAULT_EXCHANGE_MEAN, DEFAULT_EXCHANGE_SCALE)
    # Expand the IndPrior2d to the pandas dataframe format
    fixed_met_prior_df = prior_met_conc_fixed.to_dataframe("mic").rename(
        columns={"parameter": "target_id", "loc": "measurement", "scale": "error_scale"})
    fixed_met_prior_df["measurement"] = np.exp(fixed_met_prior_df["measurement"])
    fixed_exchange_prior_df = prior_exchange_fixed.to_dataframe("flux").rename(
        columns={"parameter": "target_id", "loc": "measurement", "scale": "error_scale"})
    return fixed_exchange_prior_df, fixed_met_prior_df


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
