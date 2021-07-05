"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.linalg import null_space

from .util import codify, rref

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
    NS = null_space(S).round(10)
    _, ix_free_flux = rref(NS.T)
    free_fluxes = S.columns[ix_free_flux]
    is_transport = pd.Series(["transport" in r for r in S.columns], index=S.columns)
    transports = is_transport.loc[is_transport].index
    enzymes = is_transport.loc[~is_transport].index
    return {
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "enzyme": list(enzymes),
        "transport": list(transports),
        "condition": list(measurements["condition_id"].unique()),
        "free_flux": list(free_fluxes),
        "free_enzyme": list(f for f in free_fluxes if f in enzymes),
        "free_transport": list(f for f in free_fluxes if f in transports),
        "NS": NS.tolist(),
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
    prior_transport_free = extract_prior_2d("transport", priors, coords["free_transport"], coords["condition"], 0.4, 0.01)
    prior_b_free = extract_prior_2d("b", priors, coords["free_enzyme"], coords["condition"], 0, 2)
    prior_enzyme_free = extract_prior_2d("enzyme", priors, coords["free_enzyme"], coords["condition"], 1, 0.1)
    prior_log_metabolite = extract_prior_2d("metabolite", priors, coords["metabolite"], coords["condition"], 0, 2)
    return {
        "N_metabolite": S.shape[0],
        "N_transport": len(coords["transport"]),
        "N_enzyme": len(coords["enzyme"]),
        "N_reaction": S.shape[1],
        "N_free_flux": len(coords["free_flux"]),
        "N_free_enzyme": len(coords["free_enzyme"]),
        "N_free_transport": len(coords["free_transport"]),
        "S": S.values.tolist(),
        "NS": coords["NS"],
        "ix_free_flux": [codify(coords["reaction"])[r] for r in coords["reaction"] if r in coords["free_flux"]],
        "ix_free_enzyme": [codify(coords["free_flux"])[r] for r in coords["free_flux"] if r in coords["enzyme"]],
        "ix_free_transport": [codify(coords["free_flux"])[r] for r in coords["free_flux"] if r in coords["transport"]],
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
        "enzyme_y_enzyme": measurements_by_type["enzyme"]["target_id"].map(codify(coords["enzyme"])).values.tolist(),
        "condition_y_enzyme": measurements_by_type["enzyme"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "y_metabolite": measurements_by_type["mic"]["measurement"].values.tolist(),
        "sigma_metabolite": measurements_by_type["mic"]["error_scale"].values.tolist(),
        "metabolite_y_metabolite": measurements_by_type["mic"]["target_id"].map(codify(coords["metabolite"])).values.tolist(),
        "condition_y_metabolite": measurements_by_type["mic"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        # priors
        "prior_dgf": [prior_dgf.location.values.tolist(), prior_dgf.scale.values.tolist()],
        "prior_transport_free": [prior_transport_free.location.values.tolist(), prior_transport_free.scale.values.tolist()],
        "prior_enzyme_free": [prior_enzyme_free.location.values.tolist(), prior_enzyme_free.scale.values.tolist()],
        "prior_b_free": [prior_b_free.location.values.tolist(), prior_b_free.scale.values.tolist()],
        "prior_log_metabolite": [prior_log_metabolite.location.values.tolist(), prior_log_metabolite.scale.values.tolist()],
        # config
        "likelihood": int(likelihood),
    }
