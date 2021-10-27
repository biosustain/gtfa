import itertools
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import arviz as az


def pair_plots(config, idata, display=True, save=False):

    for var in ["dgr"]:
        axes = az.plot_pair(
            idata, group="posterior",
            var_names=[var],
            coords={"condition": ["condition_1"]},
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
        if save:
            plot_dir = config.result_dir / "plots"
            plot_dir.mkdir()
            plt.savefig(plot_dir / f"{var}_pair_plot.png")
        if display:
            plt.show()


def analyse(config):
    # If we are developing we don't want to save the pair plot files
    save_plots = not config.devel
    idata = az.from_netcdf(config.result_dir / "infd.nc")
    if config.analyse.get("pair", False):
        pair_plots(config, idata, save=save_plots)
    if config.analyse.get("dens", False):
        az.plot_density(idata, var_names=["log_metabolite", "b", "flux", "log_enzyme"])
        plt.show()