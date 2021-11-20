import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy

import src.util as util
from src.fitting import stan_input_from_dir
from src.model_conversion import write_model_files
# Don't delete
from model_setup import ecoli_model, model_small
from src.pandas_to_cmdstanpy import get_free_x_and_rows, get_exchange_rxns


def test_get_free_fluxes_solution_many():
    """ Test that the free and calculated functions satisfy the steady state equation"""
    total_rows = 20
    total_cols = 100

    for nrows in range(3, total_rows):
        for ncols in range(nrows+1, total_cols):
            random_matrix = (scipy.sparse.rand(nrows, ncols, density=0.3, random_state=nrows+ncols).toarray() - 0.5) * 3
            random_matrix = np.around(random_matrix)
            get_free_fluxes_solution(random_matrix)


def get_free_fluxes_solution(random_matrix):
    """ Test that random free fluxes and calculated determined fluxes give the a steady state soltuion"""
    repeats = 10
    eps = 1e-8
    nrows, ncols = random_matrix.shape
    free_cols, fixed_mat = util.get_free_fluxes(random_matrix)
    for repeat in range(repeats):
        v = np.zeros(ncols)
        random_v = np.random.randn(sum(free_cols))
        v[free_cols] = random_v
        # Now assign the values to the rest of the fluxes
        fixed_fluxes = fixed_mat @ random_v
        v[~free_cols] = fixed_fluxes
        assert all(abs(random_matrix @ v) < eps)


@pytest.mark.usefixtures("model_small")
def test_excluded_reactions_single(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Test first without the new excluded reaction
    stan_input = stan_input_from_dir(test_dir)
    assert stan_input["N_exchange"] == 2, "Standard transport reaciton"
    model_small.Exclude_list = ["g6p/g1p"]
    # Write the files again
    write_model_files(model_small, test_dir)
    stan_input = stan_input_from_dir(test_dir)
    # Test the expected input
    assert stan_input["N_exchange"] == 3, "Expect extra transport reaction"


@pytest.mark.usefixtures("model_small")
def test_excluded_reactions_double(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Test first without the new excluded reaction
    stan_input = stan_input_from_dir(test_dir)
    assert stan_input["N_exchange"] == 2, "Standard transport reaciton"
    model_small.Exclude_list = ["g6p/g1p", "f6p/g6p"]
    # Write the files again
    write_model_files(model_small, test_dir)
    stan_input = stan_input_from_dir(test_dir)
    # Test the expected input
    assert stan_input["N_exchange"] == 4, "Expect extra transport reaction"


@pytest.mark.usefixtures("model_small")
def test_excluded_reactions_not_present(model_small):
    # Add the test dir
    test_dir = Path("test_dir")
    # Test first without the new excluded reaction
    stan_input = stan_input_from_dir(test_dir)
    assert stan_input["N_exchange"] == 2, "Standard transport reaciton"
    model_small.Exclude_list = ["not_present"]
    # Write the files again
    write_model_files(model_small, test_dir)
    stan_input = stan_input_from_dir(test_dir)
    # Test the expected input
    assert stan_input["N_exchange"] == 2, "Expect extra transport reaction"


@pytest.mark.usefixtures("model_small")
def test_free_x_calculation_true(model_small):
    test_dir = Path("test_dir")
    orders = ['EX_f1p_c', 'f6p_c']
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    free_x, _ = get_free_x_and_rows(S, orders)
    assert free_x[orders].sum() == 2, "Exchange and f6p_c are valid free x variables"


@pytest.mark.usefixtures("model_small")
def test_free_x_calculation_false(model_small):
    test_dir = Path("test_dir")
    orders = ['SK_g6p_c', 'EX_f1p_c']
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    free_x, _ = get_free_x_and_rows(S, orders)
    assert free_x[orders].sum() == 1, "Only one exchange reaction can be free"


@pytest.mark.usefixtures("model_small")
def test_free_x_shuffle_S():
    test_dir = Path("test_dir")
    orders = ['SK_g6p_c', 'EX_f1p_c']
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    for i in range(50):
        # Shuffle the columns
        np.random.seed(i)
        shuffled_S = S.loc[np.random.permutation(S.index), np.random.permutation(S.columns)]
        free_x, _ = get_free_x_and_rows(shuffled_S, orders)
        assert free_x[orders].sum() == 1, "Only one exchange reaction can be free"


@pytest.mark.usefixtures("model_small")
def test_all_small():
    """ Test all possible combinations of fixed and free in the small model"""
    test_dir = Path("test_dir")
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col="metabolite")
    exchange = get_exchange_rxns(S)
    x_vars = pd.concat([S.columns[exchange].to_series(), S.index.to_series()])
    # Now find the combinations of possible free variables
    combs = itertools.combinations(x_vars, 2)
    # Now test each of the combinations
    for comb in combs:
        comb = pd.Series(comb, index=comb)
        free_x, _ = get_free_x_and_rows(S, comb)
        # If the given variables (the final two) are free, then store them
        if all(comb.isin(['SK_g6p_c', 'EX_f1p_c'])) or all(comb.isin(['f6p_c', 'g1p_c'])):
            assert free_x[comb].sum() == 1, "Only one exchange reaction can be free"
        else:
            assert free_x[comb].sum() == 2, "All other combinations should be free"