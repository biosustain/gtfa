"""Functions for turning cmdstanpy output into arviz InferenceData objects."""

from typing import List, Dict
import pandas as pd
from .pandas_to_cmdstanpy import get_coords


def get_infd_kwargs(
    S: pd.DataFrame, measurements: pd.DataFrame, sample_kwargs: Dict
) -> Dict:
    """Get a dictionary of keyword arguments to arviz.from_cmdstanpy."""
    return dict(
        coords=get_coords(S, measurements),
        dims={
            "b_free": ["b_free_enzyme"],
            "dgf": ["metabolite"],
        },
        save_warmup=sample_kwargs["save_warmup"],
    )
