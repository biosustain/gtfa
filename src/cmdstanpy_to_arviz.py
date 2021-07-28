"""Functions for turning cmdstanpy output into arviz InferenceData objects."""

from typing import List, Dict
import pandas as pd
from .pandas_to_cmdstanpy import get_coords


def get_infd_kwargs(
    S: pd.DataFrame, measurements: pd.DataFrame, sample_kwargs: Dict
) -> Dict:
    """Get a dictionary of keyword arguments to arviz.from_cmdstanpy."""
    if "save_warmup" in sample_kwargs.keys():
        save_warmup = sample_kwargs["save_warmup"]
    else:
        save_warmup = True
    return dict(
        coords=get_coords(S, measurements),
        dims={
            "dgf": ["metabolite"],
            "dgr": ["condition", "reaction"],
            "log_metabolite": ["condition", "metabolite"],
            "b": ["condition", "enzyme_names"],
            "b_free": ["condition", "enzyme_free"],
            "enzyme": ["condition", "enzyme_names"],
            "flux": ["condition", "reaction"]
        },
        save_warmup=save_warmup,
    )
