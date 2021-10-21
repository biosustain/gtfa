"""Functions for turning cmdstanpy output into arviz InferenceData objects."""

from typing import List, Dict
import pandas as pd
from .pandas_to_cmdstanpy import get_coords, reorder_s


def get_infd_kwargs(
    S: pd.DataFrame, measurements: pd.DataFrame, sample_kwargs: Dict
) -> Dict:
    """Get a dictionary of keyword arguments to arviz.from_cmdstanpy."""
    if "save_warmup" in sample_kwargs.keys():
        save_warmup = sample_kwargs["save_warmup"]
    else:
        save_warmup = True
    return dict(
        coords=get_coords(reorder_s(S), measurements),
        dims={
            # Free parameters
            "b": ["condition", "internal_names"],
            "enzyme": ["condition", "internal_names"],
            "log_metabolite_free": ["condition", "free_met_conc"],
            "transport_free": ["condition", "free_exchange"],
            "dgf": ["metabolite"],
            # Fixed parameters
            "dgr": ["condition", "reaction"],
            "flux": ["condition", "reaction"],
            "log_metabolite": ["condition", "metabolite"],
            "x": ["condition", "x_names"]
        },
        save_warmup=save_warmup,
    )
