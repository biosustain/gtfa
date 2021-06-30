"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from .util import codify

REL_TOL = 1e-12
FUNCTION_TOL = 1e-12
MAX_NUM_STEPS = int(1e10)

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
    is_transport = pd.Series(["transport" in r for r in S.columns], index=S.columns)
    transports = is_transport.loc[is_transport].index
    enzymes = is_transport.loc[~is_transport].index
    return {
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "enzyme": list(enzymes),
        "transport": list(transports),
        "condition": list(measurements["condition_id"].unique()),
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
    log_metabolite_guess = pd.DataFrame(index=coords["condition"], columns=coords["metabolite"])
    for _, row in measurements_by_type["mic"].iterrows():
        condition, mic, y = row[["condition_id", "target_id", "measurement"]]
        log_metabolite_guess.loc[condition, mic] = np.log(y)
    prior_dgf = extract_prior_1d("dgf", priors, coords["metabolite"], -200, 200)
    prior_transport = extract_prior_2d("transport", priors, coords["transport"], coords["condition"], 0.4, 0.01)
    prior_b = extract_prior_2d("b", priors, coords["enzyme"], coords["condition"], 0, 1)
    prior_enzyme = extract_prior_2d("enzyme", priors, coords["enzyme"], coords["condition"], 1, 0.1)
    return {
        "N_metabolite": S.shape[0],
        "N_transport": len(coords["transport"]),
        "N_enzyme": len(coords["enzyme"]),
        "N_reaction": S.shape[1],
        "S": S.values.tolist(),
        "reaction_to_enzyme": [codify(coords["enzyme"])[r] if r in coords["enzyme"] else 0 for r in coords["reaction"]],
        "reaction_to_transport": [codify(coords["transport"])[r] if r in coords["transport"] else 0 for r in coords["reaction"]],
        "enzyme_to_reaction": [codify(coords["reaction"])[r] for r in coords["reaction"] if r in coords["enzyme"]],
        "transport_to_reaction": [codify(coords["reaction"])[r] for r in coords["reaction"] if r in coords["transport"]],
        "N_condition": measurements["condition_id"].nunique(),
        "N_y_enzyme": len(measurements_by_type["enzyme"]),
        "N_y_metabolite": len(measurements_by_type["mic"]),
        "N_y_flux": len(measurements_by_type["flux"]),
        "y_flux": measurements_by_type["flux"]["measurement"].values.tolist(),
        "sigma_flux": measurements_by_type["flux"]["measurement"].values.tolist(),
        "reaction_y_flux": measurements_by_type["flux"]["target_id"].map(codify(coords["reaction"])).values.tolist(),
        "condition_y_flux": measurements_by_type["flux"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "y_enzyme": measurements_by_type["enzyme"]["measurement"].values.tolist(),
        "sigma_enzyme": measurements_by_type["enzyme"]["measurement"].values.tolist(),
        "enzyme_y_enzyme": measurements_by_type["enzyme"]["target_id"].map(codify(coords["enzyme"])).values.tolist(),
        "condition_y_enzyme": measurements_by_type["enzyme"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "y_metabolite": measurements_by_type["mic"]["measurement"].values.tolist(),
        "sigma_metabolite": measurements_by_type["mic"]["measurement"].values.tolist(),
        "metabolite_y_metabolite": measurements_by_type["mic"]["target_id"].map(codify(coords["metabolite"])).values.tolist(),
        "condition_y_metabolite": measurements_by_type["mic"]["condition_id"].map(codify(coords["condition"])).values.tolist(),
        "likelihood": int(likelihood),
        "log_metabolite_guess": log_metabolite_guess.values,
        "prior_dgf": [prior_dgf.location.values.tolist(), prior_dgf.scale.values.tolist()],
        "prior_transport": [prior_transport.location.values.tolist(), prior_transport.scale.values.tolist()],
        "prior_b": [prior_b.location.values.tolist(), prior_b.scale.values.tolist()],
        "prior_enzyme": [prior_enzyme.location.values.tolist(), prior_enzyme.scale.values.tolist()],
        "rel_tol": REL_TOL,
        "function_tol": FUNCTION_TOL,
        "max_num_steps": MAX_NUM_STEPS,
    }
