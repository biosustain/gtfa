"""Get an input to cmdstanpy.CmdStanModel.sample from a pd.DataFrame."""

from dataclasses import dataclass
from typing import Dict, List
from numpy.linalg import matrix_rank
import pandas as pd

from .util import codify

REL_TOL = 1e-12
FUNCTION_TOL = 1e-9
MAX_NUM_STEPS = int(1e6)

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
    is_drain = S.abs().sum() == 1
    N_b_bound = matrix_rank(S.T)
    drains = is_drain.loc[is_drain].index
    enzymes = is_drain.loc[~is_drain].index
    b_bound_enzymes = enzymes[:N_b_bound]
    b_free_enzymes = enzymes[N_b_bound:]
    return {
        "reaction": list(S.columns),
        "metabolite": list(S.index),
        "enzyme": list(enzymes),
        "b_free_enzyme": b_free_enzymes,
        "b_bound_enzyme": b_bound_enzymes,
        "drain": list(drains),
        "condition": list(measurements["experiment_id"].unique()),
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
    y_metabolite = measurements_by_type["mic"]
    y_flux = measurements_by_type["flux"]
    y_enzyme = (
        measurements_by_type["enzyme"]
        if "enzyme" in measurements_by_type.keys()
        else pd.DataFrame([])
    )
    coords = get_coords(S, measurements)
    b_bound_guess = [[0 for _ in coords["b_bound_enzyme"]] for _ in coords["condition"]]
    prior_dgf = extract_prior_1d("dgf", priors, coords["metabolite"], -200, 200)
    prior_drain = extract_prior_2d("drain", priors, coords["drain"], coords["condition"], 0.4, 0.1)
    prior_b_free = extract_prior_2d("b_free", priors, coords["b_free_enzyme"], coords["condition"], 1, 2)
    prior_enzyme = extract_prior_2d("enzyme", priors, coords["enzyme"], coords["condition"], 1, 0.1)
    prior_metabolite = extract_prior_2d("metabolite", priors, coords["metabolite"], coords["condition"], 1, 0.1)
    return {
        "N_metabolite": S.shape[0],
        "N_drain": len(coords["drain"]),
        "N_enzyme": len(coords["enzyme"]),
        "N_reaction": S.shape[1],
        "N_b_free": len(coords["b_free_enzyme"]),
        "N_b_bound": len(coords["b_bound_enzyme"]),
        "S": S.values.tolist(),
        "ix_b_free": [codify(coords["enzyme"])[e] for e in coords["b_free_enzyme"]],
        "ix_b_bound": [codify(coords["enzyme"])[e] for e in coords["b_bound_enzyme"]],
        "reaction_to_enzyme": [codify(coords["enzyme"])[r] if r in coords["enzyme"] else 0 for r in coords["reaction"]],
        "reaction_to_drain": [codify(coords["drain"])[r] if r in coords["drain"] else 0 for r in coords["reaction"]],
        "N_condition": measurements["experiment_id"].nunique(),
        "N_y_enzyme": len(y_enzyme),
        "N_y_metabolite": len(y_metabolite),
        "N_y_flux": len(y_flux),
        "y_flux": y_flux["measurement"].values.tolist(),
        "sigma_flux": y_flux["measurement"].values.tolist(),
        "reaction_y_flux": y_flux["target_id"].map(codify(coords["reaction"])).values.tolist(),
        "condition_y_flux": y_flux["experiment_id"].map(codify(coords["condition"])).values.tolist(),
        "y_enzyme": y_enzyme["measurement"].values.tolist(),
        "sigma_enzyme": y_enzyme["measurement"].values.tolist(),
        "enzyme_y_enzyme": y_enzyme["target_id"].map(codify(coords["enzyme"])).values.tolist(),
        "condition_y_enzyme": y_enzyme["experiment_id"].map(codify(coords["condition"])).values.tolist(),
        "y_metabolite": y_metabolite["measurement"].values.tolist(),
        "sigma_metabolite": y_metabolite["measurement"].values.tolist(),
        "metabolite_y_metabolite": y_metabolite["target_id"].map(codify(coords["metabolite"])).values.tolist(),
        "condition_y_metabolite": y_metabolite["experiment_id"].map(codify(coords["condition"])).values.tolist(),
        "likelihood": int(likelihood),
        "b_bound_guess": b_bound_guess,
        "prior_dgf": [prior_dgf.location.values.tolist(), prior_dgf.scale.values.tolist()],
        "prior_drain": [prior_drain.location.values.tolist(), prior_drain.scale.values.tolist()],
        "prior_b_free": [prior_b_free.location.values.tolist(), prior_b_free.scale.values.tolist()],
        "prior_enzyme": [prior_enzyme.location.values.tolist(), prior_enzyme.scale.values.tolist()],
        "prior_metabolite": [prior_metabolite.location.values.tolist(), prior_metabolite.scale.values.tolist()],
        "rel_tol": REL_TOL,
        "function_tol": FUNCTION_TOL,
        "max_num_steps": MAX_NUM_STEPS,
    }
