"""Functions for turning cmdstanpy output into arviz InferenceData objects."""

from typing import List, Dict
import pandas as pd

from .model_configuration import ModelConfiguration
from .pandas_to_cmdstanpy import get_coords


def get_infd_kwargs(
    S: pd.DataFrame, measurements: pd.DataFrame, priors: pd.DataFrame, order, sample_kwargs: Dict
) -> Dict:
    """Get a dictionary of keyword arguments to arviz.from_cmdstanpy."""
    if "save_warmup" in sample_kwargs.keys():
        save_warmup = sample_kwargs["save_warmup"]
    else:
        save_warmup = True
    return dict(
        coords=get_coords(S, measurements, priors, order=order),
        dims={
            # Free parameters
            "b": ["condition", "internal_names"],
            "log_enzyme": ["condition", "internal_names"],
            "dgf": ["metabolite"],
            # Fixed parameters
            "dgr": ["condition", "internal_names"],
            "flux": ["condition", "reaction"],
            "log_metabolite": ["condition", "metabolite"],
            "x": ["condition", "x_names"]
        },
        save_warmup=save_warmup,
    )
