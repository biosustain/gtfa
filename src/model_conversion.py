import logging
import pathlib
from copy import deepcopy

import cobra.util.array
import multitfa.core.tmodel
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_model_files(
    tmodel: multitfa.core.tmodel, model_dir: pathlib.Path, eps=1e-4
):
    """
    Take a multitfa tmodel and write priors.csv and stoichiometry.csv files
    :return:
    """
    # Add the metabolite and reaction names
    mets = [tmet.id for tmet in tmodel.metabolites]
    rxns = [trxn.id for trxn in tmodel.reactions]
    # Make the excluded reaction into further non-thermodynamic reactions
    to_exclude = deepcopy(tmodel.Exclude_list)
    for i, rxn in enumerate(rxns):
        if rxn in tmodel.Exclude_list:
            logger.warning(
                f"Reaction {rxn} in exclude list added as exchange reaction"
            )
            rxns[i] = "EXCL_" + rxn
            to_exclude.remove(rxn)
    if to_exclude:
        logger.warning(
            f"{to_exclude} were in the model exclude list but not in the model reactions"
        )
    # Write the stoichiometric matrix
    S = cobra.util.array.create_stoichiometric_matrix(tmodel)
    s_to_write = pd.DataFrame(S)
    s_to_write.columns = rxns
    s_to_write.columns.name = "reactions"
    s_to_write.index = mets
    s_to_write.index.name = "metabolite"

    # The dgf priors
    dgf_means = [tmet.delG_f for tmet in tmodel.metabolites]
    # Vector of components for each dgf
    components = np.concatenate(
        [tmet.compound_vector.T for tmet in tmodel.metabolites], axis=1
    )
    # Transform the covariance matrix of the components to that of the individual compounds
    cov = components.T @ multitfa.util.thermo_constants.covariance @ components
    # # Sparsify the matrix a little by removing very small covariances
    # cov[np.abs(cov) < eps] = 0
    # Convert to a dataframe
    cov = pd.DataFrame(cov, columns=mets, index=mets)
    # Write the enzyme concentration priors
    num_mets = len(tmodel.metabolites)
    column_data = zip(["dgf"] * num_mets, mets, [""] * num_mets, dgf_means)
    dgf_df = pd.DataFrame(column_data)
    dgf_df.columns = ["parameter", "target_id", "condition_id", "loc"]
    dgf_df = dgf_df.set_index("parameter")
    # Write the final files
    dgf_df.to_csv(model_dir / "priors.csv")
    s_to_write.to_csv(model_dir / "stoichiometry.csv")
    cov.to_csv(model_dir / "priors_cov.csv")
    # The concentration/enzyme/exchange priors are all represented by the defaults
