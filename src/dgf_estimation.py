import logging
import pathlib

import numpy as np
import pandas as pd
from equilibrator_api import ComponentContribution
from equilibrator_cache import Compound, Q_
from equilibrator_cache.exceptions import MissingDissociationConstantsException

root_dir = pathlib.Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def calc_model_dgfs_with_prediction_error(model, cc=None):
    """
    With the given predictor, calculate the formation energies of all metabolites in the model.

    Some of this code was borrowed from the equilibrator tutorial
    https://equilibrator.readthedocs.io/en/latest/equilibrator_examples.html
    :param model:
    :return:
    """
    if cc is None:
        cc = ComponentContribution()
    compound_ids = [m.id for m in model.metabolites]
    compound_list = get_eq_compounds(model, cc)
    # A dataframe with the manually calculated group decompositions
    dgf_means, dgf_cov, met_groups, missing_estimates = get_cov_eq(cc, compound_ids, compound_list)
    # Add the legendre transform to the formation energies for the different compartments
    if hasattr(model, "compartment_conditions"):
        adjustment = calculate_legendre_transform(compound_ids, compound_list, model)
        dgf_means += adjustment
    else:
        logger.warning("No compartment conditions found in the model. The legendre transform were not applied")
    # Add the prediction error to compounds outside the training set that were estimated with the gc method
    N_cc = cc.predictor.preprocess.Nc
    MSE_gc = cc.predictor.params.MSE.loc["gc"]
    MSE_rc = cc.predictor.params.MSE.loc["rc"]
    cc_mets = met_groups[:, :N_cc].any(axis=1)
    gc_mets = met_groups[:, N_cc:].any(axis=1)
    none_mets = ~met_groups.any(axis=1)
    assert (none_mets | np.logical_xor(cc_mets, gc_mets)).all(), "The metabolites should either be in the cc or gc group"
    dgf_cov.loc[cc_mets, cc_mets] += np.diag(np.full(cc_mets.sum(), MSE_rc))
    dgf_cov.loc[gc_mets, gc_mets] += np.diag(np.full(gc_mets.sum(), MSE_gc))
    # Add the variances of the missing values
    missing_estimates = np.array(missing_estimates, dtype=int)
    mse_inf = cc.predictor.preprocess.RMSE_inf ** 2
    dgf_cov.iloc[missing_estimates, missing_estimates] = np.diag(np.full(len(missing_estimates), mse_inf))
    return dgf_means, dgf_cov


def calculate_legendre_transform(compound_ids, compound_list, model):
    adjustments = np.zeros(len(model.metabolites))
    logger.info("Adding the legendre transform to the formation energies")
    for i, m in enumerate(model.metabolites):
        cond = model.compartment_conditions.loc[m.compartment]
        # Convert to pint units
        pH, I, T, p_mg = Q_(cond["pH"]), Q_(cond["I"], "M"), Q_(cond["T"], "K"), Q_(cond["p_mg"])
        try:
            adjustments += compound_list[i].transform(pH, I, T, p_mg).m_as("kJ/mol")
        except MissingDissociationConstantsException:
            logger.warning(f"Could not adjust formation energy of {compound_ids[i]} because it does not have "
                           f"dissociation constants")
    return adjustments


def get_cov_eq(cc, compound_ids, compound_list):
    """
    Calculate the covaraince matrix of the formation energies of the model metabolites using equilibrator's own methods
    """
    compound_vectors, missing_estimates = get_group_matrix(cc, compound_ids, compound_list)
    # Using matrix multiplication instead of concatenating the individual results. It's also easier because we have
    # hand-calculated compounds
    compound_Lc = cc.predictor.preprocess.L_c @ compound_vectors.T
    compound_Linf = cc.predictor.preprocess.L_inf @ compound_vectors.T
    dgf_cov_mat = compound_Lc.T @ compound_Lc + cc.predictor.preprocess.RMSE_inf * compound_Linf.T @ compound_Linf
    dgf_means = cc.predictor.preprocess.mu @ compound_vectors.T
    # Convert to pandas
    dgf_cov_mat = pd.DataFrame(dgf_cov_mat, index=compound_ids, columns=compound_ids)
    dgf_means = pd.Series(dgf_means, index=compound_ids)
    return dgf_means, dgf_cov_mat, compound_vectors, missing_estimates


def get_group_matrix(cc, compound_ids, compound_list):
    params = cc.predictor.params
    man_group_df = pd.read_csv(root_dir / "data" / "raw" / "new_compound_groups.csv", index_col=0)
    exception_list = pd.read_csv(root_dir / "data" / "raw" / "exceptions.csv", header=None, index_col=0)
    # Numbers of
    Ng = params.dimensions.at["Ng", "number"]
    Nc = params.dimensions.at["Nc", "number"]
    n_groups = Ng + Nc
    # Get a list of groups for each compound
    groups = []
    missing_estimates = []
    for i, c in enumerate(compound_list):
        c_id = compound_ids[i]
        # This is quite fragile
        if c_id[:-2] in exception_list.index:
            logger.info(f"Compound {c_id} is in the list of exceptions and has a dgf and variance of 0")
            c_groups = np.zeros(n_groups)
        else:
            c_groups = get_groups(c, c_id, cc, man_group_df)
            if c_groups is None:
                logger.warning(f"The metabolite {c_id} could not be decomposed into groups. It is assumed to have 0 ")
                c_groups = np.zeros(n_groups)
                missing_estimates.append(i)
        groups.append(c_groups)
    # Now make a matrix of the groups (both reactant and group contribution)
    G = np.vstack(groups)
    return G, missing_estimates


def get_groups(c, c_id, cc, man_group_df):
    groups = cc.predictor.preprocess.get_compound_vector(c)
    if groups is None:
        # Try to fill in with manual group decompositions
        if c_id in man_group_df.columns:
            logger.info(f"The group decomposition of compound {c_id} was performed manually")
            groups = man_group_df.loc[:, c_id]
    return groups


def get_eq_compounds(model, cc):
    # A file containing a mapping from bigg to Kegg IDs
    ids_path = root_dir / "data" / "raw" / "from_gollub_2020" / "compound_ids.csv"
    id_df = pd.read_csv(ids_path, index_col=0, comment="#")
    # Remove duplicate index entries
    id_df = id_df.loc[~id_df.index.duplicated(keep="first")]
    compound_list = []
    for m in model.metabolites:
        bigg_id = m.id[:-2]
        kegg_compound = get_kegg_match(cc, bigg_id, id_df)
        bigg_compound = cc.get_compound(f"bigg.metabolite:{bigg_id}")
        if kegg_compound is not None and bigg_compound is not None and kegg_compound != bigg_compound:
            logger.warning(f"{bigg_id} does not match the corresponding kegg ID {kegg_compound}. The compound matching "
                           f"{kegg_compound} will be used.")
        compound_list.append(kegg_compound or bigg_compound)
    # Test for missing compounds
    missing_compounds = [i for i, c in enumerate(compound_list) if c is None]
    compound_ids = [c.id for c in model.metabolites]
    for missing_compound in missing_compounds:
        logger.warning(f"Compound {compound_ids[missing_compound]} is missing from the database")
        raise NotImplementedError("Managing missing compounds is currently not supported")
    return compound_list


def get_kegg_match(cc, bigg_id, id_df):
    # Search for a kegg annoation first
    if bigg_id in id_df.index:
        # Search through the manual annoations for metabolites
        matches = id_df.at[bigg_id, 'Kyoto Encyclopedia of Genes and Genomes']
        if pd.isna(matches):
            logger.info(f"Couldn't find KEGG match for {bigg_id}")
            return None
        if "|" in matches:
            matches = matches.split("|")
        else:
            matches = [matches]
    else:
        logger.info(f"Couldn't find KEGG match for {bigg_id}")
        return None
    # Find all compounds
    match_ids = [cc.get_compound(f"kegg:{match_id}") for match_id in matches]
    # Check that all non-None compounds are the same
    valid_matches = [match for match in match_ids if match is not None]
    if len(valid_matches) == 0:
        logger.warning(f"Could not find compound {bigg_id}")
    elif len(valid_matches) > 1:
        if not all(match == valid_matches[0] for match in valid_matches):
            logger.warning(f"Multiple conflicting matches for compound {bigg_id}. Using first match.")
    return valid_matches[0]

