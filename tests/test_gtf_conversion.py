import numpy as np
import pandas as pd
import pytest

from src.dgf_estimation import calc_model_dgfs_with_prediction_error
from tests.model_setup import build_small_test_model_exchanges


def test_small_model(model_small_rankdef_thermo):
    """
    Test that the metabolites of the small model are correctly converted (compared to values manually from the website)
    """

    dgfs, dgf_cov = calc_model_dgfs_with_prediction_error(model_small_rankdef_thermo)
    met_ids = [m.id for m in model_small_rankdef_thermo.metabolites]
    expected_dgfs = pd.Series([-515.7, -508.1, -252.5, -272.1, -2263.6, -1388.7, -1055.5], index=met_ids)
    assert np.allclose(dgfs, expected_dgfs, atol=0.1)
    # A test could be included here for the covariance matrix, but it is not clear how to compare it to the values from the website
    # expected_95_ci = pd.Series([2.3, 5.7, 6.7, 3.9, 2.9, 2.4, 1.5], index=met_ids)
    # expected_sd = expected_95_ci / 1.96
    # assert np.allclose(np.sqrt(np.diag(dgf_cov)), expected_sd, atol=0.1)


@pytest.mark.xfail(raises=NotImplementedError, reason="Duplicate compounds aren't supported yet")
def test_small_model_different_conditions():
    """
    Test that the metabolites of the small model are correctly converted for different conditions (compared to values manually from the website)
    C: pH 8, ionic strength 0.5, temperature 298.15, p MG 5.0
    P: pH 6.5, ionic strength 0.1, temperature 298.15, p Mg 0.2
    """
    model = build_small_test_model_exchanges()
    dgfs, dgf_cov = calc_model_dgfs_with_prediction_error(model)
    met_ids = [m.id for m in model.metabolites]
    expected_dgfs = pd.Series([-510.3, -502.7, -237.6, -257.2, -2225.5, -1352.6, -1052.5,  # cytosol
                               -531.2, -526.3, -283.3, -302.4, -2359.8, -1480.6, -1073.7]  # periplasm
                              , index=met_ids)
    assert np.allclose(dgfs, expected_dgfs, atol=0.1)
    # A test could be included here for the covariance matrix, but it is not clear how to compare it to the values from the website
    # expected_95_ci = pd.Series([2.3, 5.7, 6.7, 3.9, 2.9, 2.4, 1.5,2.3, 5.7, 6.7, 3.9, 2.9, 2.4, 1.5], index=met_ids)
    # expected_sd = expected_95_ci / 1.96
    # assert np.allclose(np.sqrt(np.diag(dgf_cov)), expected_sd, atol=0.1)
# Another test with non-default params
