import logging
import pathlib

import cobra.util.array
import numpy as np
import pandas as pd
import scipy

from src import util
from src.dgf_estimation import calc_model_dgfs_with_prediction_error

PROTON_INCHI = "InChI=1S/p+1"

logger = logging.getLogger(__name__)

root_dir = pathlib.Path(__file__).parent.parent


def write_gollub2020_models(mat_files: [pathlib.Path], model_dir: pathlib.Path):
    """ Read in the model files from the Gollub2020 PTA paper.

    This requires special attention because there are some extra properties that need to be accounted for"""
    mat_files = list(mat_files)
    assert len(mat_files) > 0, "No mat files provided"
    # Get the condition names
    condition_names = [path.stem.split("2015_")[1] for path in mat_files]
    dgf_means = []
    dgf_covs = []
    met_conc_means = []
    log_conc_sd = []
    stoichiometric_matrices = []
    exclude_lists = []
    for mat_file in mat_files:
        model_struct = scipy.io.loadmat(mat_file)
        # Read in the model
        model = cobra.io.mat.from_mat_struct(model_struct["model"])
        # Read in the conditions in each of the compartments
        model.compartment_conditions = get_compartment_conditions(model, model_struct)
        # Add the membrane potential correction
        model.dgr_memb_correction = model_struct["model"]["drgCorr"][0, 0].flatten()
        # Get a list of excluded reactions
        exclude_rxns = model_struct["model"]["isConstraintRxn"][0, 0].flatten() == 0
        model.exclude_list = [model.reactions[i].id for i in np.where(exclude_rxns)[0]]
        exclude_lists.append(model.exclude_list)
        # Convert to a pandas dataframe
        S_df = util.get_smat_df(model)
        stoichiometric_matrices.append(S_df)
        # Get the dgf means
        dgfs, dgf_cov = calc_model_dgfs_with_prediction_error(model)
        dgf_means.append(dgfs)
        dgf_covs.append(dgf_cov)
        # Get the log concentration means
        log_conc_means = model_struct["model"]["logConcMean"][0, 0].flatten()
        met_conc_means.append(log_conc_means)
        log_conc_cov = model_struct["model"]["logConcCov"][0, 0]
        # Check that the log conc covariance matrix is diagonal
        assert np.allclose(log_conc_cov, np.diag(np.diag(log_conc_cov))), "Log concentration covariance matrix should be diagonal"
        # Get the diagonal of the matrix
        log_conc_sd.append(np.sqrt(np.diag(log_conc_cov)))
    # Check that the dgf priors are the same for all conditions
    assert all(len(dgfs) == len(dgf_means[0]) for dgfs in dgf_means) and \
           all(np.allclose(dgf_means[0], dgf_means[i]) for i in range(1, len(dgf_means)))
    # Check that the covariance matrices are the same for all conditions
    assert all(dgfs.shape[0] == dgf_means[0].shape[0] for dgfs in dgf_means) and \
           all(np.allclose(dgf_covs[0], dgf_covs[i]) for i in range(1, len(dgf_covs)))
    # All stoichiometric matrices should be the same
    assert all(np.allclose(stoichiometric_matrices[0], stoichiometric_matrices[i]) for i in
               range(1, len(stoichiometric_matrices)))
    # All exclude lists should be the same
    assert all(np.allclose(exclude_lists[0], exclude_lists[i]) for i in range(1, len(exclude_lists)))
    # Send the files for writing
    write_model_files(model_dir, stoichiometric_matrices[0], dgf_means[0], dgf_covs[0], condition_names,
                      met_conc_means, log_conc_sd, exclude_lists[0])


def get_compartment_conditions(model, model_struct):
    compartment_conditions = pd.DataFrame(index=model.compartments, columns=["pH", "I", "T", "p_mg"], dtype=float)
    met_phs = model_struct["model"]["metsPh"][0, 0].flatten()
    met_p_mg = model_struct["model"]["metsPhi"][0, 0].flatten()
    met_is = model_struct["model"]["metsI"][0, 0].flatten()
    met_ts = model_struct["model"]["metsT"][0, 0].flatten()
    for i, m in enumerate(model.metabolites):
        met_conditions = pd.Series(index=["pH", "I", "T", "p_mg"], data=[met_phs[i], met_is[i], met_ts[i], met_p_mg[i]],
                                   dtype=float)
        assert compartment_conditions.loc[m.compartment].isnull().all() or \
               np.allclose(compartment_conditions.loc[m.compartment], met_conditions), \
            "Metabolites in the same compartment should have the same conditions"
        compartment_conditions.loc[m.compartment] = met_conditions
    return compartment_conditions

def make_dgf_df(met_ids, dgf_means):
    # Write the enzyme concentration priors
    num_mets = len(met_ids)
    cols = ["parameter", "target_id", "condition_id", "loc", "scale"]
    column_data = zip(["dgf"] * num_mets, met_ids, [""] * num_mets, dgf_means, [""] * num_mets)
    dgf_df = pd.DataFrame(column_data, columns=cols)
    return dgf_df


def make_met_conc_df(met_ids, log_conc_means, log_met_sd, condition_name):
    # Check the input
    if all((log_conc_means >= 0) & (log_conc_means <= 1)):
        assert False, "All log concentrations are between 0 and 1, are you sure they are not actually concentrations?"
    # Write the enzyme concentration priors
    num_mets = len(log_conc_means)
    cols = ["measurement_type", "target_id", "condition_id", "measurement", "error_scale"]
    column_data = zip(["mic"] * num_mets, met_ids, [condition_name] * num_mets, log_conc_means, log_met_sd)
    dgf_df = pd.DataFrame(column_data, columns=cols)
    return dgf_df


def write_model_files(model_dir, S, dgf_means, dgf_cov_mat, conditions=None, log_conc_means=None, log_met_sd=None, exclude_list=None):
    # Check for excluded reactions
    if exclude_list is not None:
        S = modify_excluded(S, exclude_list)
    met_ids = S.index
    cov = pd.DataFrame(dgf_cov_mat, columns=met_ids, index=met_ids)
    dgf_df = make_dgf_df(met_ids, dgf_means)
    dgf_df.to_csv(model_dir / "priors.csv", index=False)
    S.to_csv(model_dir / "stoichiometry.csv")
    cov.to_csv(model_dir / "priors_cov.csv")
    if log_conc_means is not None:
        assert log_met_sd is not None and conditions is not None, "Means, conditions and std devations should be present for log metabolites"
        assert len(conditions) == len(log_conc_means) == len(
            log_met_sd), "The lengths of conditions, log_met_means and " \
                         "log_met_sd should be the same"
        all_measurements = []
        for i, condition in enumerate(conditions):
            assert len(log_met_sd[i].shape) == 1, "The log_met_sd should be a vector"
            log_met_df = make_met_conc_df(met_ids, log_conc_means[i], log_met_sd[i], condition)
            all_measurements.append(log_met_df)
        all_measurements = pd.concat(all_measurements)
        all_measurements.to_csv(model_dir / "measurements.csv", index=False)


def modify_excluded(S, excluded_reactions):
    new_S = S.copy()
    new_rxns = []
    for rxn in S.columns:
        if rxn in excluded_reactions:
            new_rxns.append("EXCL_" + rxn)
        else:
            new_rxns.append(rxn)
    new_S.columns = new_rxns
    return new_S