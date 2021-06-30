import pytest
import scipy
import numpy as np
import src.util as util

def test_get_free_fluxes_solution_many():
    """ Test that the free and calculated functions satisfy the steady state equation"""
    total_rows = 20
    total_cols = 20
    for nrows in range(3, total_rows):
        for ncols in range(nrows+1, total_cols):
            random_matrix = (scipy.sparse.rand(nrows, ncols, density=0.3, random_state=nrows+ncols).toarray() - 0.5) * 3
            random_matrix = np.around(random_matrix)
            test_get_free_fluxes_solution(random_matrix)

def test_get_free_fluxes_solution(random_matrix):
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