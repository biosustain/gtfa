import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import arviz as az

NC_FILE_PRIOR = os.path.join("results", "infd", "infd_toy_prior.nc")
NC_FILE_POSTERIOR = os.path.join("results", "infd", "infd_toy_likelihood.nc")

def get_individual_samples(idata_prior, n, condition=0, chain=0):
    cols = ["dgr", "flux", "dgf", "b", "enzyme", "log_metabolite"]
    all_cols = []
    for col in cols:
        param_runs = getattr(idata_prior.posterior, col).values
        if len(param_runs.shape) == 3:
            # DGF
            values = param_runs[chain, :n, :]
        else:
            values = param_runs[chain, :n, condition, :]
        colnames = [col + str(i) for i in range(values.shape[1])]
        all_cols.append(pd.DataFrame(values, columns=colnames))
    return pd.concat(all_cols, axis=1)

def main():
    idata_prior = az.from_netcdf(NC_FILE_PRIOR)

    a = get_individual_samples(idata_prior, 50)
    a.to_csv("temp_sample.csv")
    # idata_posterior = az.from_netcdf(NC_FILE_POSTERIOR)

    for var in ["dgr", "flux", "dgf", "b", "log_metabolite"]:
        axes = az.plot_pair(
            idata_prior, group="posterior",
            var_names=[var],
            coords={"condition": ["condition_1"]},
            divergences=True
        )
        # Center the values for the dgr pair plots
        if var == "dgr":
            for ax in axes.flatten():
                yabs_max = abs(max(ax.get_ylim(), key=abs))
                xabs_max = abs(max(ax.get_xlim(), key=abs))
                ax.set_xlim([-xabs_max, xabs_max])
                ax.set_ylim([-yabs_max, yabs_max])
        plt.savefig(f"results/plots/{var}_pair_plot_prior.png")
        # az.plot_pair(
        #     idata_posterior,
        #     group="posterior",
        #     var_names=[var],
        #     coords={"condition": "condition_1", "reaction": ["A", "B", "C", "D", "E"]},
        #     divergences=True
        # )
        # plt.savefig(f"results/plots/{var}_pair_plot.png")


if __name__ == "__main__":
    main()
