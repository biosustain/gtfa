"""Some handy python functions."""
import logging
from typing import Iterable, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import norm


def codify(l: Iterable[str]) -> Dict[str, int]:
    return {l_i: i + 1 for i, l_i in enumerate(l)}


def one_encode(s: pd.Series) -> pd.Series:
    """Replace a series's values with 1-indexed integer factors.

    :param s: a pandas Series that you want to factorise.

    """
    return pd.Series(pd.factorize(s)[0] + 1, index=s.index)


def make_columns_lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame's columns lower case.

    :param df: a pandas DataFrame
    """
    new = df.copy()
    new.columns = [c.lower() for c in new.columns]
    return new


def get_lognormal_params_from_qs(
    x1: float, x2: float, p1: float, p2: float
) -> Tuple[float, float]:
    """Find parameters for a lognormal distribution from two quantiles.

    i.e. get mu and sigma such that if X ~ lognormal(mu, sigma), then pr(X <
    x1) = p1 and pr(X < x2) = p2.

    :param x1: the lower value
    :param x2: the higher value
    :param p1: the lower quantile
    :param p1: the higher quantile

    """
    logx1 = np.log(x1)
    logx2 = np.log(x2)
    denom = norm.ppf(p2) - norm.ppf(p1)
    sigma = (logx2 - logx1) / denom
    mu = (logx1 * norm.ppf(p2) - logx2 * norm.ppf(p1)) / denom
    return mu, sigma


def get_normal_params_from_qs(
    x1: float, x2: float, p1: float, p2: float
) -> Tuple[float, float]:
    """find parameters for a normal distribution from two quantiles.

    i.e. get mu and sigma such that if x ~ normal(mu, sigma), then pr(x <
    x1) = p1 and pr(x < x2) = p2.

    :param x1: the lower value
    :param x2: the higher value
    :param p1: the lower quantile
    :param p1: the higher quantile

    """
    denom = norm.ppf(p2) - norm.ppf(p1)
    sigma = (x2 - x1) / denom
    mu = (x1 * norm.ppf(p2) - x2 * norm.ppf(p1)) / denom
    return mu, sigma


def get_99_pct_params_ln(x1: float, x2: float):
    """Wrapper assuming you want the 0.5%-99.5% inter-quantile range.

    :param x1: the lower value such that pr(X > x1) = 0.005
    :param x2: the higher value such that pr(X < x2) = 0.995

    """
    return get_lognormal_params_from_qs(x1, x2, 0.005, 0.995)


def get_99_pct_params_n(x1: float, x2: float):
    """Wrapper assuming you want the 0.5%-99.5% inter-quantile range.

    :param x1: the lower value such that pr(X > x1) = 0.005
    :param x2: the higher value such that pr(X < x2) = 0.995

    """
    return get_normal_params_from_qs(x1, x2, 0.005, 0.995)

def rref(A: np.ndarray, tol: float=1.0e-12):
    """
    Calculate the reduced row echelon form of a matrix.

    Taken from https://stackoverflow.com/questions/7664246/python-built-in-function-to-do-matrix-reduction
    :param A: The input matrix
    :param tol:
    :return: M: The matrix in row echelon form
             jb: ?
    """
    M = A.copy() # We don't want to modify it in place
    m, n = M.shape
    i, j = 0, 0
    jb = []
    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(M[i:m, j])) + i
        p = np.abs(M[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            M[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                M[[i, k], j:n] = M[[k, i], j:n]
            # Divide the pivot row i by the pivot element M[i, j]
            M[i, j:n] = M[i, j:n] / M[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    M[k, j:n] -= M[k, j] * M[i, j:n]
            i += 1
            j += 1
    return M, jb

def get_free_fluxes(S: np.ndarray):
    """
    Calculate the set of free fluxes from a stoichiometric matrix. Also returns a set of vectors for each dependent
    reaction for its calculation from the
    :param S: Stoichiometric matrix
    :return: free_fluxes: A binary mask of the free fluxes
             fixed_fluxes: A matrix containing the vectors required to calculate each dependent flux.

    e.g.
        M = np.array([[1, 0,-1, 2], [0, 1, -1, 2], [-1, 0, 2, -2]])
        free_cols, fixed_mat = get_free_fluxes(M)
        v = np.zeros(M.shape[0])
        random_v = np.random.randn(sum(free_cols))
        v[free_cols] = random_v
        # Now assign the values to the rest of the fluxes
        fixed_fluxes = fixed_mat @ random_v
        v[~free_cols] = fixed_fluxes
        M @ v # Confirm output vector is all near 0
    """
    nrows, ncols = S.shape
    rr_mat, jb = rref(S)
    assert all([x == y for x,y in zip(jb, sorted(jb))]), "Dealing with column rearrangements is not yet implemented"
    fixed_fluxes = np.full(ncols, False)
    for i in range(nrows):
        nz = np.nonzero(rr_mat[i, :])[0]
        if len(nz) == 0:
            break
        # The pivot is the first nonzero element of the row
        fixed_fluxes[nz[0]] = True
    free_fluxes = ~fixed_fluxes
    if not any(free_fluxes):
        raise RuntimeError("No free fluxes detected")
    # Now to get the equations for the fixed fluxes from the free fluxes
    num_fixed = fixed_fluxes.sum()
    fixed_fluxes = rr_mat[:num_fixed, free_fluxes] * -1
    return free_fluxes, fixed_fluxes

