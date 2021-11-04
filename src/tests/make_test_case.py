"""
Functions for making a consistent dataset with fixed and free variables as is expected in our dataset.
"""
import logging
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from src.pandas_to_cmdstanpy import reorder_s
from src.util import get_free_fluxes

RT = 0.008314 * 298.15

logger = logging.getLogger(__name__)


def namevec(name, vec):
    return [f"{name}_{i}" for i in range(len(vec))]


def calc_internal_fluxes(s_gamma, e, b, dgf, c):
    """From a set of parameters, calculate the fluxes"""
    dgr = s_gamma.T @ (dgf + RT * np.log(c))
    return dgr.multiply(b * e, axis=0)


def get_s_x(S, b, e, exchange_rxns):
    """Get the modified s matrix for calculating the free and fixed fluxes"""
    n_exchange = exchange_rxns.sum()
    n_mets, n_rxns = S.shape
    s_x = np.zeros((n_rxns, n_exchange + n_mets))
    s_x[:n_exchange, :n_exchange] = np.identity(n_exchange)
    s_x[n_exchange:, n_exchange:] = S.loc[:, ~exchange_rxns].T.mul(
        b * e, axis=0
    )
    return s_x


def get_s_c(S, b, e, exchange_rxns):
    s_x = get_s_x(S, b, e, exchange_rxns)
    return S.values @ s_x


def calc_fixed(S, b, e, c_free, t_free, dgf, free_vars):
    """
    Calculate all fixed parameters from the free parameters
    """
    num_mets, num_rxns = S.shape
    exchange_rxns = S.columns.str.contains("SK_") | S.columns.str.contains(
        "EX_"
    )
    # Check that they are at the start of the reactions
    assert ~any(
        exchange_rxns[exchange_rxns.sum() :]
    ), "All reactions should be first"
    # Determine the s_c and s_x matrices
    s_c = get_s_c(S, b, e, exchange_rxns)
    s_x = get_s_x(S, b, e, exchange_rxns)
    # More useful numbers
    num_exchange = exchange_rxns.sum()
    num_x = num_exchange + num_mets
    # Define some masks for the different parts of the x vector
    conc_x = np.full(num_x, False)
    conc_x[num_exchange:] = True
    free_c_mask = free_vars[conc_x]
    fixed_c_mask = ~free_vars[conc_x]
    # Calculate the rhs of the equation (from the free vars)
    x = np.full(num_x, np.NAN)
    assert (
        len(c_free) == free_c_mask.sum()
    ), "The number of free c must be correct"
    assert (
        len(t_free) == free_vars[~conc_x].sum()
    ), "The number of free t must be correct"
    x[conc_x & free_vars] = dgf[free_c_mask] + RT * c_free
    x[~conc_x & free_vars] = t_free
    rhs = -s_c[:, free_vars] @ x[free_vars]
    # Determine the corresponding fixed variables
    x[~free_vars] = np.linalg.solve(s_c[:, ~free_vars], rhs)
    # Back-calculate all the fixed variables
    c = np.zeros(num_mets)
    c[free_c_mask] = c_free  # The concentration vars of the fixed variables
    c[fixed_c_mask] = (x[~free_vars & conc_x] - dgf[fixed_c_mask]) / RT
    # Calculate the fluxes
    # Exchange fluxes
    v = s_x @ x
    check_fluxes(S, b, c, conc_x, dgf, e, exchange_rxns, num_rxns, s_c, s_x, x)
    return v, c


def check_fluxes(S, b, c, conc_x, dgf, e, exchange_rxns, num_rxns, s_c, s_x, x):
    # Check the s_x matrix
    assert all(
        S @ s_x @ x < 1e-10
    ), "All conc changes should be approximately 0"
    # Check the s_c matrix
    assert all(s_c @ x < 1e-10), "All conc changes should be approximately 0"
    # Check the standard calculation
    test_v = np.zeros(num_rxns)
    dgr = S.T[~exchange_rxns] @ (dgf + RT * c)
    test_v[~exchange_rxns] = dgr * b * e
    test_v[exchange_rxns] = x[~conc_x]
    assert all(S @ test_v < 1e-10)


def find_params(test_dir):
    """Make a dataframe filled with samples of model parameters that have reasonable values"""
    # Now write the measurements to file
    result_dir = test_dir / "results"
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col=0)
    S = reorder_s(S)
    exchange_rxns = S.columns.str.contains("SK_") | S.columns.str.contains(
        "EX_"
    )
    # Get the free and fixed fluxes
    n_internal = (~exchange_rxns).sum()
    exchange_rxns = S.columns.str.contains("SK_") | S.columns.str.contains(
        "EX_"
    )
    s_c = get_s_c(S, np.ones(n_internal), np.ones(n_internal), exchange_rxns)
    free_vars, _ = get_free_fluxes(np.flip(s_c, axis=1))
    free_vars = np.flip(free_vars)
    dgf = pd.read_csv(test_dir / "priors.csv", index_col=1)["loc"]
    exchange_rxns = S.columns.str.contains("SK_") | S.columns.str.contains(
        "EX_"
    )
    n_internal = (~exchange_rxns).sum()
    params = []
    for i in range(1000):
        c_free = np.exp(np.random.randn(1) * 2 - 8)
        t_free = np.array([1])
        b = np.exp(np.random.randn(n_internal) * 3 + 3)
        e = np.exp(np.random.randn(n_internal) * 2 - 8)
        v, c = calc_fixed(S, b, e, np.log(c_free), t_free, dgf, free_vars)
        dgr = S.loc[:, ~exchange_rxns].T @ (dgf + RT * c)
        c_range = (c > -11) & (c < -5)
        b_range = (np.log(b) > -4) & (np.log(b) < 8)
        e_range = (np.log(e) > -11) & (np.log(e) < -5)
        if all(c_range) and all(b_range) & all(e_range):
            param = chain.from_iterable([dgf, c, b, e, v, dgr])
            params.append(param)
    columns = chain.from_iterable(
        [
            namevec("dgf", dgf),
            namevec("c", c),
            namevec("b", b),
            namevec("e", e),
            namevec("v", v),
            namevec("dgr", dgr),
        ]
    )
    return pd.DataFrame(params, columns=list(columns))


if __name__ == "__main__":
    "./data/fake/simulation_study"
    test_dir = Path(sys.argv[1])
    if not test_dir.exists():
        logger.error("The given directory doesn't exist")
        sys.exit()
    reasonable_samples = find_params(test_dir)
    reasonable_samples.to_csv(test_dir / "samples.csv")
