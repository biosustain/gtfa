"""Functions for fitting models using cmdstanpy."""
import itertools
import os
import re
import warnings
from typing import List
import arviz as az
import numpy as np
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
import pandas as pd
from .model_configuration import ModelConfiguration
from .pandas_to_cmdstanpy import get_stan_input
from .cmdstanpy_to_arviz import get_infd_kwargs

# Location of this file
HERE = os.path.dirname(os.path.abspath(__file__))

# Where to save files
LOO_DIR = os.path.join(HERE, "..", "results", "loo")
SAMPLES_DIR = os.path.join(HERE, "..", "results", "samples")
INFD_DIR = os.path.join(HERE, "..", "results", "infd")
JSON_DIR = os.path.join(HERE, "..", "results", "input_data_json")


def generate_samples(model_config: ModelConfiguration) -> None:
    """Run cmdstanpy.CmdStanModel.sample, do diagnostics and save results.

    :param study_name: a string
    """
    print(f"Fitting model configuration {model_config.name}...")
    infd_file = os.path.join(INFD_DIR, f"infd_{model_config.name}.nc")
    json_file = os.path.join(JSON_DIR, f"input_data_{model_config.name}.json")
    priors = pd.read_csv(os.path.join(model_config.data_folder, "priors.csv"))
    measurements = pd.read_csv(os.path.join(model_config.data_folder, "measurements.csv"))
    S = pd.read_csv(os.path.join(model_config.data_folder, "stoichiometry.csv"), index_col="metabolite")
    likelihood = model_config.likelihood
    stan_input = get_stan_input(measurements, S, priors, likelihood)
    print(f"Writing input data to {json_file}")
    jsondump(json_file, stan_input)
    model = CmdStanModel(
        model_name=model_config.name, stan_file=model_config.stan_file
    )
    print(f"Writing csv files to {SAMPLES_DIR}...")
    mcmc = model.sample(
        data=stan_input,
        output_dir=SAMPLES_DIR,
        **model_config.sample_kwargs,
    )
    print(mcmc.diagnose().replace("\n\n", "\n"))
    infd_kwargs = get_infd_kwargs(S, measurements, model_config.sample_kwargs)
    df = to_dataframe(mcmc, infd_kwargs["dims"], infd_kwargs["coords"])
    dgr_scatter(df, 1)
    dgr_scatter(df, 2)
    check_df(df, infd_kwargs["coords"], S)
    infd = az.from_cmdstan(
        mcmc.runset.csv_files, **infd_kwargs
    )

    print(az.summary(infd))
    print(f"Writing inference data to {infd_file}")
    infd.to_netcdf(infd_file)

def dgr_scatter(df, chain):
    import matplotlib.pyplot as plt
    axes = pd.plotting.scatter_matrix(df[(0, "dgr", f"condition_{chain}")].loc[:, "A":"E"])
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if i == j:
                continue
            ax = axes[i,j]
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            xabs_max = abs(max(ax.get_xlim(), key=abs))
            # Draw horizontal and vertical dotted lines at 0
            ax.plot((0,0), (-yabs_max,yabs_max), linestyle="--", linewidth=0.5, color="black")
            ax.plot((-xabs_max, xabs_max), (0, 0), linestyle="--", linewidth=0.5, color="black")
            ax.set_xlim([-xabs_max, xabs_max])
            ax.set_ylim([-yabs_max, yabs_max])
    plt.suptitle(f"Chain {chain}")
    plt.show()


# def calc_things(b, e, log_met, flux_stan, dgr_stan, dgf_stan):
#     R = 0.008314
#     T = 298.15
#     S = pd.read_csv("/home/jason/Documents/Uni/thesis/gtfa/data/raw/toy_model/stoichiometry.csv")
#     S = S.set_index("metabolite")
#     num_met, num_rxn = S.shape
#     num_trans = 2
#     transport_rxns = S.columns.str.contains("transport")
#     s_mod = S.copy()
#     s_mod.loc[:, ~transport_rxns] = s_mod.loc[:, ~transport_rxns] * b * e
#
#     s_gamma = S.T[~transport_rxns]
#     s_gamma_mod = np.zeros((num_rxn, num_met + num_trans))
#     s_gamma_mod[:num_trans, :num_trans] = np.identity(num_trans)
#     s_gamma_mod[num_trans:, num_trans:] = s_gamma.to_numpy()
#     s_total = s_mod @ s_gamma_mod
#     import src.util as util
#     free, transform = util.get_free_fluxes(s_total.to_numpy())
#     # rhs
#     free_x = log_met[2:]
#     rhs = -s_total.loc[:, free].to_numpy() @ free_x.to_numpy()
#     fixed_x = np.linalg.solve(s_total.loc[:, ~free], rhs)
#     all_x = np.zeros(s_total.shape[1])
#     all_x[free] = free_x
#     all_x[~free] = fixed_x
#     trans_x = all_x[:num_trans]
#     therm_x = all_x[num_trans:]
#     dgr = S.T @ therm_x
#     fluxes = np.zeros(num_rxn)
#     fluxes[transport_rxns] = trans_x
#     fluxes[~transport_rxns] = dgr[~transport_rxns] * b * e
#     conc_change = S @ fluxes
#     print(conc_change)


def check_df(df, coords, S, eps=1e-2):
    """
    Check the samples for consistency
    :param df: pandas df from to_dataframe
    :param coords: the set of coordinate names
    """
    R = 0.008314
    T = 298.15
    chains = df.columns.unique(level="chain")
    # The list of actual conditions. These should be present for all params
    conditions = df.columns.unique(level="cond").drop("shared")

    # chain = chains[0]
    # condition = conditions[1]

    for chain in chains:
        # The basics - b free and b match
        for condition in conditions:
            bs = df.loc[:, (chain, "b", condition)]
            # Stan just doesn't return a param if it is size 0
            if "transport_free" in df.columns.unique(1):
                transport_free = df.loc[:, (chain, "transport_free", condition)]
            else:
                transport_free = []
            # The internal and transport free fluxes should be present in the b vector
            # assert (bs[coords["transport_free"]] == transport_free).all(axis=None)
            dgrs = df.loc[:, (chain, "dgr", condition)]
            dgfs = df.loc[:, (chain, "dgf", "shared")]
            log_conc = df.loc[:, (chain, "log_metabolite", condition)]
            pred_dgr = S.T @ (dgfs + R * T * log_conc).T
            assert ((dgrs.T - pred_dgr) < eps).all(axis=None), "dgrs should match"
        # Check the steady state assumpiton
        for condition in conditions:
            fluxes = df[(chain, "flux", condition)]
            conc_change = S @ fluxes.T
            failed = conc_change > eps
            if failed.any(axis=None):
                failed_inds = failed.columns[failed.any()].tolist()
                warnings.warn(f"Samples {failed_inds} in chain {chain} in {condition} do not sastify the steady state constraint.")
        # Check the directions of reactions
        for condition in conditions:
            dgrs = df.loc[:, (chain, "dgr", condition, coords["enzyme_names"])]
            fluxes = df.loc[:, (chain, "flux", condition, coords["enzyme_names"])]
            bs = df.loc[:, (chain, "b", condition, coords["enzyme_names"])]
            assert not (dgrs * bs * fluxes < 0).any(axis=None)




def to_dataframe(mcmc, dims, coords):
    table = mcmc.draws()
    num_chains = table.shape[1]
    param_dfs = []
    for param, cols in mcmc.stan_vars_cols.items():
        if not param in dims:
            # Skip internal params
            continue
        if len(dims[param]) == 1:
            param_dims = [["shared"], coords[dims[param][0]]]
        else:
            param_dims = [coords[dim] for dim in dims[param]]
        # Get a list of column tuples for the multiindex
        # Add the chains
        column_lists = [range(num_chains), [param]]
        column_lists.extend(param_dims[::-1]) # Needs to be reversed because mcmc expands in the other order
        column_tuples = list(itertools.product(*column_lists))
        # Now put all the chains side-by-side
        chain_tables = [table[:, chain, cols] for chain in range(num_chains)]
        all_chains = np.concatenate(chain_tables, axis=1)
        param_df = pd.DataFrame(all_chains, columns=pd.MultiIndex.from_tuples(column_tuples, names=["chain", "param", "ind", "cond"]))
        param_dfs.append(param_df)
    df = pd.concat(param_dfs, axis=1)
    # Reorder the levels to have the conditions then indices
    df.columns = df.columns.reorder_levels([0, 1, 3, 2])
    return df
