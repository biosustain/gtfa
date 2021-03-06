import itertools
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scipy

import src.util as util
from src.fitting import run_stan
from src.model_configuration import load_model_configuration
from src.pandas_to_cmdstanpy import get_free_x_and_rows, get_exchange_rxns, get_coords


def test_get_free_fluxes_solution_many():
    """ Test that the free and calculated functions satisfy the steady state equation"""
    total_rows = 20
    total_cols = 100

    for nrows in range(3, total_rows):
        for ncols in range(nrows + 1, total_cols):
            random_matrix = (scipy.sparse.rand(nrows, ncols, density=0.3,
                                               random_state=nrows + ncols).toarray() - 0.5) * 3
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


def test_free_x_calculation_true(model_small):
    temp_dir = Path("temp_dir")
    orders = ['EX_f1p_c', 'f6p_c']
    S = pd.read_csv(temp_dir / "stoichiometry.csv", index_col="metabolite")
    free_x, _ = get_free_x_and_rows(S, orders)
    assert free_x[orders].sum() == 2, "Exchange and f6p_c are valid free x variables"


def test_free_x_calculation_false(model_small):
    temp_dir = Path("temp_dir")
    orders = ['SK_g6p_c', 'EX_f1p_c']
    S = pd.read_csv(temp_dir / "stoichiometry.csv", index_col="metabolite")
    free_x, _ = get_free_x_and_rows(S, orders)
    assert free_x[orders].sum() == 1, "Only one exchange reaction can be free"


def test_free_x_shuffle_S(model_small):
    model_dir = Path("temp_dir")
    orders = ['SK_g6p_c', 'EX_f1p_c']
    S = pd.read_csv(model_dir / "stoichiometry.csv", index_col="metabolite")
    for i in range(50):
        # Shuffle the columns
        np.random.seed(i)
        shuffled_S = S.loc[np.random.permutation(S.index), np.random.permutation(S.columns)]
        free_x, _ = get_free_x_and_rows(shuffled_S, orders)
        assert free_x[orders].sum() == 1, "Only one exchange reaction can be free"


def test_all_small(model_small):
    """ Test all possible combinations of fixed and free in the small model"""
    model_dir = Path("temp_dir")
    S = pd.read_csv(model_dir / "stoichiometry.csv", index_col="metabolite")
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


# @pytest.mark.xfail(reason="This test is failing because the directionalities are wrong")
def test_directionality(small_model_irreversible):
    """Test to make sure that irreversible reactions are going in the right direction"""
    config = load_model_configuration("test_small_likelihood_full.toml")
    # This will be cleaned up with the rest of the files
    config.result_dir = config.data_folder / "results"
    # Remove the results file
    mcmc = run_stan(config)
    # Now test the results
    S = pd.read_csv(config.data_folder / "stoichiometry.csv", index_col=0)
    exchange_rxns = get_exchange_rxns(S)
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    priors = pd.read_csv(config.data_folder / "priors.csv")
    data = az.from_cmdstanpy(mcmc, coords=get_coords(S, measurements, priors, config.order))
    flux_df = data.posterior.flux.to_dataframe().unstack("flux_dim_0").loc[:, ~exchange_rxns]
    b_df = data.posterior.b.to_dataframe().unstack("b_dim_0")
    dgr_df = data.posterior.dgr.to_dataframe().unstack("dgr_dim_0")
    # Check that the irreversible fluxes are going in the right direction
    assert ((flux_df.values * b_df.values * dgr_df.values) < 0).all(axis=None)
    # Check the absolute fluxes of reactions that are very irreversible
    assert (flux_df.iloc[:, 0] > 0).all()  # g6p/g1p
    assert (flux_df.iloc[:, 1] > 0).all()  # g1p/f1p
    assert (flux_df.iloc[:, 2] < 0).all()  # f1p/f6p
    assert (flux_df.iloc[:, 3] < 0).all()  # f6p/g6p
