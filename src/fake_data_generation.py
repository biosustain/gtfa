"""Provides a function for generating fake data. Work in progress!"""
import itertools
import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from matplotlib import pyplot as plt

from src import util
from src.cmdstanpy_to_arviz import get_infd_kwargs
from src.fitting import stan_input_from_dir, stan_input_from_config
from src.model_configuration import load_model_configuration, ModelConfiguration

import arviz as az

# # Indexing
# "ix_free_met_to_free": make_index_map(coords, "free_x_names", ["free_met_conc"]),
# "ix_free_ex_to_free": make_index_map(coords, "free_x_names", ["free_exchange"]),
# "ix_fixed_met_to_fixed": make_index_map(coords, "fixed_x_names", ["fixed_met_conc"]),
# "ix_fixed_ex_to_fixed": make_index_map(coords, "fixed_x_names", ["fixed_exchange"]),
# "ix_free_to_x": make_index_map(coords, "x_names", ["free_x_names"]),
# "ix_fixed_to_x": make_index_map(coords, "x_names", ["fixed_x_names"]),
# "ix_ex_to_x": make_index_map(coords, "x_names", ["exchange"]),
# "ix_met_to_x": make_index_map(coords, "x_names", ["metabolite"]),
# "ix_internal_to_rxn": make_index_map(coords, "reaction", ["internal_names"]),
# "ix_ex_to_rxn": make_index_map(coords, "reaction", ["exchange"]),
# "ix_free_met_to_met": make_index_map(coords, "metabolite", ["free_met_conc"]),
# "ix_fixed_met_to_met": make_index_map(coords, "metabolite", ["fixed_met_conc"]),
# "ix_free_ex_to_ex": make_index_map(coords, "exchange", ["exchange", "free_x_names"]),
# "ix_fixed_ex_to_ex": make_index_map(coords, "exchange", ["exchange", "fixed_x_names"]),
# "ix_free_row_to_met": make_index_map(coords, "metabolite", ["free_rows"]),
# "ix_fixed_row_to_met": make_index_map(coords, "metabolite", ["fixed_rows"]),
# # measurements
# "N_condition": len(coords["condition"]),
# "N_y_enzyme": len(measurements_by_type["enzyme"]),
# "N_y_metabolite": len(measurements_by_type["mic"]),
# "N_y_flux": len(measurements_by_type["flux"]),
# "y_flux": measurements_by_type["flux"]["measurement"].values.tolist(),
# "sigma_flux": measurements_by_type["flux"]["error_scale"].values.tolist(),
# "reaction_y_flux": measurements_by_type["flux"]["target_id"].map(codify(coords["reaction"])).values.tolist(),
# "condition_y_flux": measurements_by_type["flux"]["condition_id"].map(
#     codify(coords["condition"])).values.tolist(),
# # Concentrations given on a log scale
# "y_enzyme": measurements_by_type["enzyme"]["measurement"].values.tolist(),
# "sigma_enzyme": measurements_by_type["enzyme"]["error_scale"].values.tolist(),
# "internal_y_enzyme": measurements_by_type["enzyme"]["target_id"].map(
#     codify(coords["internal_names"])).values.tolist(),
# "condition_y_enzyme": measurements_by_type["enzyme"]["condition_id"].map(
#     codify(coords["condition"])).values.tolist(),
# # Concentrations given on a log scale
# "y_metabolite": measurements_by_type["mic"]["measurement"].values.tolist(),
# "sigma_metabolite": measurements_by_type["mic"]["error_scale"].values.tolist(),
# "metabolite_y_metabolite": measurements_by_type["mic"]["target_id"].map(
#     codify(coords["metabolite"])).values.tolist(),
# "condition_y_metabolite": measurements_by_type["mic"]["condition_id"].map(
#     codify(coords["condition"])).values.tolist(),
# # priors
# "prior_dgf_mean": prior_dgf_mean.values.tolist(),
# "prior_dgf_cov": priors_cov.values.tolist(),
# "prior_exchange_free": [prior_exchange_free.location.values.tolist(),
#                         prior_exchange_free.scale.values.tolist()],
# "prior_enzyme": [prior_enzyme.location.values.tolist(), prior_enzyme.scale.values.tolist()],
# "prior_b": [prior_b.location.values.tolist(), prior_b.scale.values.tolist()],
# "prior_free_met_conc": [prior_met_conc_free.location.values.tolist(),
#                         prior_met_conc_free.scale.values.tolist()],
from src.pandas_to_cmdstanpy import get_coords_condition_list
from src.util import ind_to_mask

DEFAULT_MET_MEASUREMENT_ERROR = 0.2
DEFAULT_ENZYME_MEASUREMENT_ERROR = 0.2
DEFAULT_FLUX_MEASUREMENT_ERROR = 0.5
RT = 0.008314 * 298.15

base_dir = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def generate_data(stan_input: dict, S_df: pd.DataFrame, samples_per_param=100, num_conditions=10, seed=42,
                  bounds=None) -> pd.DataFrame:
    """Simulate data from the stan input dictionary"""
    # Set the numpy seed
    np.random.seed(seed)
    # Get the coordinates of the various dimensions
    conditions = ["condition_" + str(i) for i in range(num_conditions)]
    coords = get_coords_condition_list(S_df, conditions)
    S = S_df.values
    nmet, nrxn, nex = stan_input["N_metabolite"], stan_input["N_reaction"], stan_input["N_exchange"]
    nx = nmet + nex
    exchange_rxns = np.array(stan_input["ix_ex_to_rxn"])
    fixed_x, free_x = np.array(stan_input["ix_fixed_to_x"]), np.array(stan_input["ix_free_to_x"])
    exchange_x, met_x = np.array(stan_input["ix_ex_to_x"]), np.array(stan_input["ix_met_to_x"])
    fixed_met, free_met = np.array(stan_input["ix_fixed_met_to_met"]), np.array(stan_input["ix_free_met_to_met"])
    free_row = np.array(stan_input["ix_free_row_to_met"])
    # Convert to masks
    fixed_x_mask, free_x_mask = ind_to_mask(fixed_x - 1, nx), ind_to_mask(free_x - 1, nx)
    exchange_x_mask, internal_x_mask = ind_to_mask(exchange_x - 1, nx), ind_to_mask(met_x - 1, nx)
    fixed_met_mask, free_met_mask = ind_to_mask(fixed_met - 1, nmet), ind_to_mask(free_met - 1, nmet)
    exchange_rxn_mask = ind_to_mask(exchange_rxns - 1, nrxn)
    free_row_mask = ind_to_mask(free_row - 1, nmet)
    S_v_base = np.zeros((nrxn, nex + nmet))
    exchange_entries = np.logical_and.outer(exchange_rxn_mask, exchange_x_mask)
    S_v_base[exchange_entries] = np.identity(nex).flatten()  # Needs to be 1d for some reason
    # The negative here ensures that negative dgr produces positive fluxes
    internal_entries = np.logical_and.outer(~exchange_rxn_mask, ~exchange_x_mask)
    S_v_base[internal_entries] = -S[:, ~exchange_rxn_mask].T.flatten()
    # Generate the parameter sets for each condition
    parameter_measurements = []
    num_params_sampled = 0
    logger.info("Generating parameter samples")
    # The dgf params are shared across all conditions and must be sampled first
    dgf = np.random.multivariate_normal(stan_input["prior_dgf_mean"], stan_input["prior_dgf_cov"],
                                        check_valid="raise")
    while num_params_sampled < num_conditions:
        logger.info(f"{num_params_sampled} of {num_conditions} conditions sampled")
        b, exchange_free, log_enzyme, log_met_conc_free = sample_free_params(stan_input)
        # Generate the data
        # Make the S_m matrix for solving the system of equations
        dgr, flux, log_met_conc = determine_fixed_params(S, S_v_base, b, dgf, exchange_free, exchange_rxn_mask,
                                                         exchange_x_mask, fixed_met_mask, fixed_x_mask, free_met_mask,
                                                         free_x_mask, log_enzyme, log_met_conc_free, free_row_mask,
                                                         nmet,
                                                         nx)
        param_dict = {"b": b, "dgr": dgr, "flux": flux, "log_met_conc": log_met_conc, "log_enzyme": log_enzyme,
                      "dgf": dgf}
        if bounds is not None and check_out_of_bounds(bounds, param_dict):
            continue
        # Sample measurements from these parameters
        enzyme_measurements, flux_measurements, log_met_conc_measurements = sample_measurements(flux, log_enzyme,
                                                                                                log_met_conc,
                                                                                                samples_per_param)
        # Make the measurement dataframes
        flux_measurement_df = make_measurement_df(flux_measurements, coords["reaction_ind"], "flux")
        log_met_measurement_df = make_measurement_df(log_met_conc_measurements, coords["metabolite_ind"], "mic")
        enzyme_measurement_df = make_measurement_df(enzyme_measurements, coords["internal_names"], "enzyme")
        # Combine them together
        measurement_df = pd.concat([flux_measurement_df, log_met_measurement_df, enzyme_measurement_df], axis=1)
        # Add the parameters
        param_df, param_index = make_param_df(b, coords, dgf, dgr, flux, log_enzyme, log_met_conc)
        # Now combine the two
        repeated_params = pd.concat([param_df] * samples_per_param)
        repeated_params.index = measurement_df.index
        measurement_df[param_index] = repeated_params
        parameter_measurements.append(measurement_df)
        num_params_sampled += 1
    # Combine the parameter sets into a single dataframe
    full_measurements_df = pd.concat(parameter_measurements, axis=0)
    return full_measurements_df


def check_out_of_bounds(var_bounds: dict, param_dict: dict):
    # Put all the bounds in the right format
    for k, v in var_bounds.items():
        # This should be iterable with elements that are tuples of length 2
        try:
            right_length = len(v[0]) == 2
            not_a_list = False
        except TypeError:
            right_length = len(v) == 2
            not_a_list = True
        if not right_length:
            raise ValueError(f"Bounds for {k} must be tuples of length 2")
        if not_a_list:
            var_bounds[k] = [v]
    # Check that the samples are within the bounds
    for bound_var, bounds in var_bounds.items():
        var_vals = param_dict[bound_var]
        in_bounds = np.full(var_vals.shape, False)
        for bound in bounds:
            # Find values that fit within bounds
            in_bounds = in_bounds | (var_vals > bound[0]) & (var_vals < bound[1])
        if not np.all(in_bounds):
            logger.info(
                f"Sample rejected because {bound_var} with values {var_vals} was out of bounds: {bounds}")
            return True
    return False


def make_param_df(b, coords, dgf, dgr, flux, log_enzyme, log_met_conc):
    param_flux_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["flux"]), coords["reaction_ind"]))
    param_dgr_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["dgr"]), coords["internal_names"]))
    param_b_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["b"]), coords["internal_names"]))
    param_log_enzyme_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["enzyme"]), coords["internal_names"]))
    param_dgf_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["dgf"]), coords["metabolite_ind"]))
    param_met_conc_tuples = list(zip(itertools.cycle(["P"]), itertools.cycle(["mic"]), coords["metabolite_ind"]))
    param_index = pd.MultiIndex.from_tuples(param_flux_tuples + param_dgr_tuples + param_b_tuples +
                                            param_log_enzyme_tuples + param_dgf_tuples + param_met_conc_tuples,
                                            names=["type", "measurement_type", "target_id"])
    param_df = pd.DataFrame(np.concatenate([flux, dgr, b, log_enzyme, dgf, log_met_conc]), index=param_index).T
    return param_df, param_index


def make_measurement_df(measurements, coords, name):
    flux_tuples = zip(itertools.cycle(["M"]), itertools.cycle([name]), coords)
    flux_index = pd.MultiIndex.from_tuples(flux_tuples, names=["type", "measurement_type", "target_id"])
    flux_measurement_df = pd.DataFrame(measurements, columns=flux_index)
    return flux_measurement_df


def sample_measurements(flux, log_enzyme, log_met_conc, samples_per_param):
    log_met_conc_measurements = np.random.normal(log_met_conc, np.full((samples_per_param, len(log_met_conc)),
                                                                       DEFAULT_MET_MEASUREMENT_ERROR))
    flux_measurements = np.random.normal(flux, np.full((samples_per_param, len(flux)), DEFAULT_FLUX_MEASUREMENT_ERROR))
    enzyme_measurements = np.random.normal(log_enzyme, np.full((samples_per_param, len(log_enzyme)),
                                                               DEFAULT_ENZYME_MEASUREMENT_ERROR))
    return enzyme_measurements, flux_measurements, log_met_conc_measurements


# NOTE: Could probably be refactored
def determine_fixed_params(S, S_v_base, b, dgf, exchange_free, exchange_rxn_mask, exchange_x_mask, fixed_met_mask,
                           fixed_x_mask, free_met_mask, free_x_mask, log_enzyme, log_met_conc_free, free_row_mask, nmet,
                           nx):
    S_v = S_v_base.copy()
    # Elementwise multiply by be
    S_v[~exchange_rxn_mask] *= (np.exp(log_enzyme[:, np.newaxis]) * np.exp(b[:,
                                                                           np.newaxis]))  # These need to be column vectors for numpy broadcasting
    S_m = S @ S_v
    x = np.zeros(nx)
    x[free_x_mask & ~exchange_x_mask] = dgf[free_met_mask] + RT * log_met_conc_free
    x[free_x_mask & exchange_x_mask] = exchange_free
    # Now solve for the fixed values
    rhs = -S_m[free_row_mask][:, free_x_mask] @ x[free_x_mask]
    x[fixed_x_mask] = np.linalg.solve(S_m[free_row_mask][:, fixed_x_mask], rhs)
    # Calculate the rest of the parameters
    flux = S_v @ x
    # Check that the steady-state solution holds
    assert np.allclose(S @ flux, 0, atol=1e-4) and np.allclose(S_m @ x, 0, atol=1e-4), "Steady state should hold"
    log_met_conc = np.zeros(nmet)
    log_met_conc[free_met_mask] = log_met_conc_free
    log_met_conc[fixed_met_mask] = (x[fixed_x_mask & ~exchange_x_mask] - dgf[fixed_met_mask]) / RT
    dgr = S[:, ~exchange_rxn_mask].T @ (dgf + RT * log_met_conc)
    assert (dgr * flux[~exchange_rxn_mask] <= 0).all(), "Fluxes should have the opposite sign to the dgr"
    assert np.allclose(flux[~exchange_rxn_mask],
                       -dgr * np.exp(b) * np.exp(log_enzyme),
                       atol=1e-4), "Internal fluxes should be be -dgr * b * log_enzyme"
    return dgr, flux, log_met_conc


def sample_free_params(stan_input, scale_divisor=5):
    # Because we want a parameter where all values correspond to the priors we need to reduce the variance so that SOME
    # samples are accepted
    # Assumes a single condition
    # NOTE: It might be worth checking here that the priors across all conditions are the same
    free_exchange_ix = np.array(stan_input["ix_free_ex_to_ex"], dtype=int) - 1
    free_exchange_rxn_ix = np.array(stan_input["ix_ex_to_rxn"])[free_exchange_ix] - 1
    free_exchange_prior_mean = np.array(stan_input["prior_exchange"][0][0])[free_exchange_rxn_ix]
    free_exchange_prior_scale = np.array(stan_input["prior_exchange"][1][0])[free_exchange_rxn_ix] / scale_divisor
    exchange_free = np.random.normal(free_exchange_prior_mean, free_exchange_prior_scale)
    b = np.random.normal(stan_input["prior_b"][0][0], np.array(stan_input["prior_b"][1][0]) / scale_divisor)
    log_enzyme = np.random.normal(stan_input["prior_enzyme"][0][0],
                                  np.array(stan_input["prior_enzyme"][1][0]) / scale_divisor)
    free_mets = np.array(stan_input["ix_free_met_to_met"], dtype=int) - 1
    free_mets_prior_mean = np.array(stan_input["prior_met_conc"][0][0])[free_mets]
    free_mets_prior_scale = np.array(stan_input["prior_met_conc"][1][0])[free_mets] / scale_divisor
    log_met_conc_free = np.random.normal(free_mets_prior_mean, free_mets_prior_scale)
    return b, exchange_free, log_enzyme, log_met_conc_free


def generate_data_and_config(config_path: Path, samples_per_param=10, num_conditions=10, measured_params=None,
                             bounds=None, cache=True):
    """
    Generate fake data for a given existing configuration using the priors.
    """
    if measured_params is None:
        measured_params = ["all"]
    config_dir = config_path.parent
    base_config = load_model_configuration(config_path)
    new_config = switch_config(base_config, measured_params, num_conditions, samples_per_param)
    # If we are caching we don't want to overwrite the generated data
    skip_sampling = True
    if cache:
        if not new_config.data_folder.exists():
            logger.info("No existing cached data. Running data generation"
                        )
            skip_sampling = False
    else:
        if new_config.data_folder.exists():
            logger.info("Removing existing data folder")
            shutil.rmtree(new_config.data_folder)
        skip_sampling = False
    if not skip_sampling:
        make_data_folder(base_config, new_config)
        S = pd.read_csv(new_config.data_folder / "stoichiometry.csv", index_col=0)
        # Generate the data
        # Load the stan input with the original data (we won't use any of the measurements, just the priors)
        stan_input = stan_input_from_config(new_config)
        data_df = generate_data(stan_input, S, samples_per_param, num_conditions, bounds=bounds)
        data_df.to_csv(new_config.data_folder / "true_params_and_measurements.csv")
        # Save the parameters as well
        data_df["P"].loc[0].to_csv(
            new_config.data_folder / "true_params.csv")  # Only need the first instance of each param
        # Select only the measured columns
        write_measurements(data_df, measured_params, new_config.data_folder, num_conditions, samples_per_param)
        # Write new config file
        (config_dir / "synthetic").mkdir(exist_ok=True)
        new_config.to_toml(config_dir / "synthetic" / (new_config.name + ".toml"))
    else:
        # read in the data df - the measurements have already been written
        logger.info("Using cached true params")
        # This has a 3 level multiarray
        data_df = pd.read_csv(new_config.data_folder / "true_params_and_measurements.csv", header=[0, 1, 2], index_col=0)
    return new_config, data_df


def make_data_folder(base_config, new_config):
    new_config.data_folder.mkdir(parents=True)
    # Copy the data folder across
    for f in base_config.data_folder.iterdir():
        if not f.is_dir():
            shutil.copy(f, new_config.data_folder)


def switch_config(base_config, measured_params, num_conditions, samples_per_param):
    # Deterine the new directories
    new_config = base_config.copy()
    new_config.name = make_model_name(measured_params, new_config, num_conditions, samples_per_param)
    new_config.data_folder = base_dir / "data" / "fake" / new_config.name
    new_config.result_dir = base_dir / "results" / new_config.name
    return new_config


def write_measurements(data_df, measured_params, data_folder, num_conditions, samples_per_param):
    if not measured_params[0] == "all":
        data_df = data_df.loc[:, data_df.columns.isin(measured_params)]
    # Convert the dataframe into the measurement format (only select  measurements)
    measurement_df = data_df["M"].stack(data_df["M"].columns.names).rename("measurement").reset_index()
    measurement_df.loc[measurement_df["measurement_type"] == "mic", "error_scale"] = DEFAULT_MET_MEASUREMENT_ERROR
    measurement_df.loc[measurement_df["measurement_type"] == "enzyme", "error_scale"] = DEFAULT_ENZYME_MEASUREMENT_ERROR
    measurement_df.loc[measurement_df["measurement_type"] == "flux", "error_scale"] = DEFAULT_FLUX_MEASUREMENT_ERROR
    # Add the conditions. We have num_per_param * n_cols entries per condition
    measurement_df["condition_id"] = [f"condition_{i + 1}" for i in range(num_conditions) for _ in
                                      range(samples_per_param * data_df["M"].shape[1])]
    # Now write the file
    measurement_df[["measurement_type", "target_id", "condition_id", "measurement", "error_scale"]].to_csv(
        data_folder / "measurements.csv", index=False)


def make_model_name(measured_params, config, num_conditions, samples_per_param):
    return f"{config.name}_{num_conditions}_{samples_per_param}_{'_'.join(measured_params)}"


if __name__ == "__main__":
    bounds = {"met_conc": (-12, -4),
              "flux": (-20, 20),
              "b": (0, 8000)}
    generate_data_and_config(
        Path("/home/jason/Documents/Uni/thesis/gtfa/model_configurations/toy_likelihood_conc_single.toml"),
        samples_per_param=3, num_conditions=1, bounds=bounds)
