"""
Functions for testing the consistency of draws from the posterior.
"""
import itertools
import warnings

import numpy as np
import pandas as pd


def check_df(df, coords, S, eps=1e-2):
    """
    Check the samples for consistency
    :param df: pandas df from to_dataframe
    :param coords: the set of coordinate names
    """
    R = 0.008314
    T = 298.15
    chains = df.columns.unique(level="chain")
    # The list of actual conditions. These should be present for all params
    conditions = df.columns.unique(level="cond").drop("shared")
    for chain in chains:
        # The basics - b free and b match
        for condition in conditions:
            bs = df.loc[:, (chain, "b", condition)]
            # Stan just doesn't return a param if it is size 0
            if "transport_free" in df.columns.unique(1):
                transport_free = df.loc[:, (chain, "transport_free", condition)]
            else:
                transport_free = []
            # The internal and transport free fluxes should be present in the b vector
            # assert (bs[coords["transport_free"]] == transport_free).all(axis=None)
            dgrs = df.loc[:, (chain, "dgr", condition)]
            dgf0s = df.loc[:, (chain, "dgf0", "shared")]
            log_conc = df.loc[:, (chain, "log_metabolite", condition)]
            pred_dgr = S.T @ (dgf0s + R * T * log_conc).T
            assert ((dgrs.T - pred_dgr) < eps).all(axis=None), "dgrs should match"
        # Check the steady state assumpiton
        for condition in conditions:
            fluxes = df[(chain, "flux", condition)]
            conc_change = S @ fluxes.T
            failed = conc_change > eps
            if failed.any(axis=None):
                failed_inds = failed.columns[failed.any()].tolist()
                warnings.warn(f"Samples {failed_inds} in chain {chain} in {condition} do not sastify the steady state constraint.")
        # Check the directions of reactions
        for condition in conditions:
            dgrs = df.loc[:, (chain, "dgr", condition, coords["enzyme_names"])]
            fluxes = df.loc[:, (chain, "flux", condition, coords["enzyme_names"])]
            bs = df.loc[:, (chain, "b", condition, coords["enzyme_names"])]
            assert not (dgrs * bs * fluxes < 0).any(axis=None)