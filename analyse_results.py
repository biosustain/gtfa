import os

from matplotlib import pyplot as plt
import arviz as az

NC_FILE_PRIOR = os.path.join("results", "infd", "infd_toy_prior.nc")
NC_FILE_POSTERIOR = os.path.join("results", "infd", "infd_toy_likelihood.nc")


def main():
    idata_prior = az.from_netcdf(NC_FILE_PRIOR)
    idata_posterior = az.from_netcdf(NC_FILE_POSTERIOR)
    for var in ["dgr", "flux", "dgf", "b_free", "enzyme_free", "log_metabolite"]:
        az.plot_pair(
            idata_prior, group="posterior",
            var_names=[var],
            coords={"condition": "condition_1", "reaction": ["A", "B", "C", "D", "E"]},
            divergences=True
        )
        plt.savefig(f"results/plots/{var}_pair_plot_prior.png")
        az.plot_pair(
            idata_posterior,
            group="posterior",
            var_names=[var],
            coords={"condition": "condition_1", "reaction": ["A", "B", "C", "D", "E"]},
            divergences=True
        )
        plt.savefig(f"results/plots/{var}_pair_plot.png")


if __name__ == "__main__":
    main()
