import logging
import pathlib
from copy import deepcopy

import cobra.util.array
from cobra import Model, Reaction, Metabolite
from equilibrator_api import ComponentContribution
import numpy as np
import pandas as pd

PROTON_INCHI = "InChI=1S/p+1"

logger = logging.getLogger(__name__)


def write_model_files(met_model: Model, model_dir: pathlib.Path, eps=1e-4, cc=None):
    """
    Take a multitfa tmodel and write priors.csv and stoichiometry.csv files
    :return:
    """
    # Add the metabolite and reaction names
    mets = [tmet.id for tmet in met_model.metabolites]
    rxns = [trxn.id for trxn in met_model.reactions]
    # Write the stoichiometric matrix
    S = cobra.util.array.create_stoichiometric_matrix(met_model)
    s_to_write = pd.DataFrame(S)
    s_to_write.columns = rxns
    s_to_write.columns.name = "reactions"
    s_to_write.index = mets
    s_to_write.index.name = "metabolite"
    # Calculate the means and covariance matrix of the dgfs
    dgf_means, dgf_cov = calc_model_dgfs(met_model, cc=cc)
    # Convert to a dataframe
    dgf_cov = pd.DataFrame(dgf_cov, columns=mets, index=mets)
    # Write the enzyme concentration priors
    num_mets = len(met_model.metabolites)
    column_data = zip(["dgf"] * num_mets, mets, [""] * num_mets, dgf_means)
    dgf_df = pd.DataFrame(column_data)
    dgf_df.columns = ["parameter", "target_id", "condition_id", "loc"]
    dgf_df = dgf_df.set_index("parameter")
    # Write the final files
    dgf_df.to_csv(model_dir / "priors.csv")
    s_to_write.to_csv(model_dir / "stoichiometry.csv")
    dgf_cov.to_csv(model_dir / "priors_cov.csv")
    # The concentration/enzyme/exchange priors are all represented by the defaults


def calc_model_dgfs(model, cc=None):
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
    standard_dgf_mu, sigmas_fin, sigmas_inf = zip(*map(cc.standard_dg_formation, compound_list))
    standard_dgf_mu, sigmas_fin, sigmas_inf = np.array(standard_dgf_mu, dtype=float), np.array(sigmas_fin), \
                                              np.array(sigmas_inf)
    # Account for protons
    proton_inds = [i for i, m in enumerate(compound_list) if m.inchi == PROTON_INCHI]
    missing_inds = [i for i, m in enumerate(standard_dgf_mu) if np.isnan(m)]
    assert np.array_equal(proton_inds, missing_inds), "Estimation of proton formation " \
                                                         "energies should not be supported"
    # Any of the non-missing values can tell us the number of groups
    num_fin_groups = sigmas_fin[~np.isnan(standard_dgf_mu)][0].shape[0]
    num_inf_groups = sigmas_inf[~np.isnan(standard_dgf_mu)][0].shape[0]
    for missing_ind in missing_inds:
        standard_dgf_mu[missing_ind] = 0
        sigmas_inf[missing_ind] = np.zeros(num_inf_groups)
        sigmas_fin[missing_ind] = np.zeros(num_fin_groups)
    # Transform into matrices
    sigmas_inf = np.vstack(sigmas_inf)
    sigmas_fin = np.vstack(sigmas_fin)
    # we now apply the Legendre transform to convert from the standard ΔGf to the standard ΔG'f
    delta_dgf_list = np.array([
        cpd.transform(cc.p_h, cc.ionic_strength, cc.temperature, cc.p_mg).m_as("kJ/mol")
        for cpd in compound_list
    ])
    standard_dgf_prime_mu = standard_dgf_mu + delta_dgf_list
    # to create the formation energy covariance matrix, we need to combine the two outputs
    # sigma_fin and sigma_inf
    standard_dgf_cov = sigmas_fin @ sigmas_fin.T + 1e6 * sigmas_inf @ sigmas_inf.T
    return pd.Series(standard_dgf_prime_mu, index=compound_ids), pd.DataFrame(standard_dgf_cov, index=compound_ids,
                                                                              columns=compound_ids)


def get_eq_compounds(model, cc):
    compound_names = ["_".join(cname.id.split("_")[:-1]) for cname in model.metabolites]
    compound_list = [cc.get_compound(f"bigg.metabolite:{cname}") for cname in compound_names]
    # Read in the IDs and use the kegg IDs if the BIGG IDs can not be found
    ids_path = pathlib.Path(__file__).parent.parent / "data" / "raw" / "from_gollub_2020" / "compound_ids.csv"
    id_df = pd.read_csv(ids_path, index_col=0).drop_duplicates()
    for i, compound in enumerate(compound_list):
        if compound is not None:
            continue
        name = compound_names[i]
        # Search for a kegg annoation first
        anns = model.metabolites[i].annotation
        if "kegg.compound" in anns:
            # The model contains a kegg annoation for the metabolite
            matches = [anns['kegg.compound']]
        elif name in id_df.index:
            # Search through the manual annoations for metabolites
            matches = id_df.at[name, 'Kyoto Encyclopedia of Genes and Genomes']
            if "|" in matches:
                matches = matches.split("|")
            else:
                matches = [matches]
        else:
            assert False, f"Could not find a kegg ID for {name}"
        # Find all compounds
        match_ids = [cc.get_compound(f"kegg:{match_id}") for match_id in matches]
        # Check that all non-None compounds are the same
        valid_matches = [match for match in match_ids if match is not None]
        if len(valid_matches) == 0:
            logger.warning(f"Could not find compound {name}")
        elif len(valid_matches) > 1:
            if not all(match == valid_matches[0] for match in valid_matches):
                logger.warning(f"Multiple conflicting matches for compound {name}. Using first match.")
            compound_list[i] = valid_matches[0]
        else:
            compound_list[i] = valid_matches[0]
    return compound_list
