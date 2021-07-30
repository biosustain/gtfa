import itertools
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import arviz as az

NC_FILE_PRIOR = os.path.join("results", "infd", "infd_toy_prior.nc")
NC_FILE_POSTERIOR = os.path.join("results", "infd", "infd_toy_likelihood.nc")


def main():
    idata_prior = az.from_netcdf(NC_FILE_PRIOR)
    idata_posterior = az.from_netcdf(NC_FILE_POSTERIOR)
    # Load the draws
    # for var in ["dgr", "flux", "dgf", "b", "log_metabolite"]:
    for var in ["dgr"]:
        axes = az.plot_pair(
            idata_prior, group="posterior",
            var_names=[var],
            coords={"condition": ["condition_1"], "reaction": ["A", "B", "C", "D", "E"]},
            divergences=True
        )
        # Center the values for the dgr pair plots
        if var == "dgr":
            for ax in axes.flatten():
                yabs_max = abs(max(ax.get_ylim(), key=abs))
                xabs_max = abs(max(ax.get_xlim(), key=abs))
                ax.plot((0, 0), (-yabs_max, yabs_max), linestyle="--", linewidth=0.5, color="black")
                ax.plot((-xabs_max, xabs_max), (0, 0), linestyle="--", linewidth=0.5, color="black")
                ax.set_xlim([-xabs_max, xabs_max])
                ax.set_ylim([-yabs_max, yabs_max])
        plt.show()
        plt.savefig(f"results/plots/{var}_pair_plot_prior.png")
        axes = az.plot_pair(
            idata_posterior,
            group="posterior",
            var_names=[var],
            coords={"condition": "condition_1", "reaction": ["A", "B", "C", "D", "E"]},
            divergences=True
        )
        if var == "dgr":
            for ax in axes.flatten():
                yabs_max = abs(max(ax.get_ylim(), key=abs))
                xabs_max = abs(max(ax.get_xlim(), key=abs))
                ax.plot((0, 0), (-yabs_max, yabs_max), linestyle="--", linewidth=0.5, color="black")
                ax.plot((-xabs_max, xabs_max), (0, 0), linestyle="--", linewidth=0.5, color="black")
                ax.set_xlim([-xabs_max, xabs_max])
                ax.set_ylim([-yabs_max, yabs_max])
        plt.show()
        plt.savefig(f"results/plots/{var}_pair_plot.png")


# def to_dataframe(mcmc, dims, coords):
#     table = mcmc.draws()
#     num_chains = table.shape[1]
#     param_dfs = []
#     for param, cols in mcmc.stan_vars_cols.items():
#         if not param in dims:
#             # Skip internal params
#             continue
#         if len(dims[param]) == 1:
#             param_dims = [["shared"], coords[dims[param][0]]]
#         else:
#             param_dims = [coords[dim] for dim in dims[param]]
#         # Get a list of column tuples for the multiindex
#         # Add the chains
#         column_lists = [range(num_chains), [param]]
#         column_lists.extend(param_dims[::-1]) # Needs to be reversed because mcmc expands in the other order
#         column_tuples = list(itertools.product(*column_lists))
#         # Now put all the chains side-by-side
#         chain_tables = [table[:, chain, cols] for chain in range(num_chains)]
#         all_chains = np.concatenate(chain_tables, axis=1)
#         param_df = pd.DataFrame(all_chains, columns=pd.MultiIndex.from_tuples(column_tuples, names=["chain", "param", "ind", "cond"]))
#         param_dfs.append(param_df)
#     df = pd.concat(param_dfs, axis=1)
#     # Reorder the levels to have the conditions then indices
#     df.columns = df.columns.reorder_levels([0, 1, 3, 2])
#     return df


if __name__ == "__main__":
    main()
