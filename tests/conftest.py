import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from cobra.io import load_model

from src import model_conversion
from src.dgf_estimation import calc_model_dgfs_with_prediction_error
from src.util import get_smat_df
from tests.model_setup import gen_random_log_concs, build_small_test_model, build_small_test_model_rankdef_stoich, \
    build_small_test_model_rankdef_thermo

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
# This makes the tests run faster
base_dir = Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    temp_dir = Path(base_dir / "tests" / "temp_dir").absolute()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def ecoli_model():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = load_model("e_coli_core")
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = calc_model_dgfs_with_prediction_error(model)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    result_dir = temp_dir / "results"
    result_dir.mkdir(parents=True)
    # Get the dgfs
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = calc_model_dgfs_with_prediction_error(model)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", -2, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small_rankdef():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model_rankdef_stoich()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = model_conversion.calc_model_dgfs_with_prediction_error(model)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", -2, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_small_rankdef_thermo():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model_rankdef_thermo()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = model_conversion.calc_model_dgfs_with_prediction_error(model)
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    result_dir = temp_dir / "results"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    result_dir.mkdir()
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    # We need at least one measurment
    header = pd.DataFrame(columns=["measurement_type", "target_id", "condition_id", "measurement", "error_scale"],
                          data=[["mic", "f6p_c", "condition_1", -2, 0.1]])
    header.to_csv(temp_dir / "measurements.csv", index=False)
    yield model
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_model_irreversible():
    # Make sure that logging print statements still work
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Write the files
    model = build_small_test_model()
    log_concs, log_conc_scales = gen_random_log_concs(model)
    dgfs, dgf_covs = calc_model_dgfs_with_prediction_error(model)
    # Change the dgfs to be irreversible
    dgfs[1] += 500  # The input has high energy
    dgfs[2] -= 500  # The output has low energy
    S = get_smat_df(model)
    temp_dir = Path("temp_dir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    result_dir = temp_dir / "results"
    result_dir.mkdir(parents=True)
    # Get the dgfs
    model_conversion.write_model_files(temp_dir, S, dgfs, dgf_covs, ["1"], [log_concs], [log_conc_scales])
    yield model
    # Clean up
    shutil.rmtree(temp_dir)
