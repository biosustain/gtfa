import logging
import shutil
from pathlib import Path

import arviz as az
import numpy as np
from matplotlib import pyplot as plt

from src.fake_data_generation import generate_data_and_config
from src.fitting import generate_samples

logger = logging.getLogger(__name__)


def sample_and_show(config_path: Path, samples_per_param=3, num_conditions=1, bounds=None, cache=True):
    if bounds is None:
        bounds = {"log_met_conc": (-13, -3),
                  "flux": [(-100, -1), (1, 100)],
                  "b": [(0, 200000)],
                  "log_enzyme": (-13, -3),
                  "dgr": (-100, 100)}
    # Generate simulated data from the config
    new_config, sim_data = generate_data_and_config(config_path, samples_per_param=samples_per_param,
                                                    num_conditions=num_conditions, bounds=bounds, cache=cache)
    perform_sampling = False
    if cache:
        if not (new_config.result_dir / "infd.nc").exists():
            logger.info("No samples found, generating new samples")
            perform_sampling = True
    else:
        perform_sampling = True
    if perform_sampling:
        # Remake the results dir
        if new_config.result_dir.exists():
            shutil.rmtree(new_config.result_dir)
            logger.info("Removed existing results dir")
        new_config.result_dir.mkdir()
        generate_samples(new_config)
    else:
        logger.info("Using cached samples")
    # Display the simulated samples
    data = az.from_netcdf(new_config.result_dir / "infd.nc")
    true_params = sim_data.loc[[0], "P"]  # Only the first sample of each param set is necessary
    rename = {"log_metabolite": "mic",
              "log_enzyme": "enzyme"}  # Convert between stan data names and generated data names
    vars_to_plot = ["dgf", "flux", "log_metabolite", "log_enzyme", "b", "dgr"]
    plot_density_true_params(data, num_conditions, rename, sim_data, true_params, vars_to_plot)
    plt.show()
    # Pair plots
    plot_pairs_true_params(data, rename, true_params, ["dgf", "log_metabolite"], 1)
    plt.show()
    plot_pairs_true_params(data, rename, true_params, ["dgf", "log_metabolite"], 2)
    plt.show()
    plot_pairs_true_params(data, rename, true_params, ["dgr", "flux"], 1)
    plt.show()
    plot_pairs_true_params(data, rename, true_params, ["dgr", "dgf"], 1)
    plt.show()


def plot_pairs_true_params(data, rename, true_params, vars_to_plot, condition_num, coords={}):
    """
    Regular pair plots but with lines showing the true parameters
    """
    axs = az.plot_pair(data, var_names=vars_to_plot, coords={"condition": f"condition_{condition_num}"})
    sim_names = [rename.get(v, v) for v in vars_to_plot]
    true_vals = true_params[sim_names].iloc[condition_num-1]
    for i in range(len(axs)):
        for j in range(i+1):
            axs[i][j].axvline(true_vals.iloc[j], color="black", linestyle="--")
            # The first row corresponds to the second entry (no need to plot the diagonal)
            axs[i][j].axhline(true_vals.iloc[i+1], color="black", linestyle="--")
    plt.show()

# NOTE: Can refactor to remove sim_data
def plot_density_true_params(data, num_conditions, rename, sim_data, true_params, vars_to_plot):
    n_params_to_plot = len(true_params[[rename.get(v, v) for v in vars_to_plot]].columns)
    total_num_plots = n_params_to_plot * num_conditions
    # Remove the extra dgf plots if it is to be plotted
    if "dgf" in vars_to_plot:
        total_num_plots -= (num_conditions - 1) * len(true_params["dgf"].columns)
    ncols = 5
    nrows = int(np.ceil(total_num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2))
    # Add the true parameters to the diagram
    axes = axes.flatten()
    axis_num = 0
    for var in vars_to_plot:
        sim_var = rename.get(var, var)
        param_names = true_params.loc[0, sim_var].columns
        if var == "dgf":
            num_plots = len(param_names)
        else:
            num_plots = len(param_names) * num_conditions
        az.plot_density(data, var_names=var, ax=axes.flatten()[axis_num:axis_num + num_plots])
        # Note: there's probably a more elegant way to do this - no time, sorry.
        if var == "dgf":
            for p in param_names:
                # Assumes ordering is maintained
                true_val = true_params[sim_var, p].iloc[0]
                axes[axis_num].axvline(true_val, color="black", linestyle="--")
                axis_num += 1
        else:
            for cond_ind in range(num_conditions):
                for p in param_names:
                    # Assumes ordering is maintained
                    true_val = true_params[sim_var, p].iloc[cond_ind]
                    axes[axis_num].axvline(true_val, color="black", linestyle="--")
                    axis_num += 1
    plt.tight_layout()
    plt.show()


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
        az.plot_density(idata, var_names=["log_metabolite", "b", "flux", "log_enzyme", "dgf", "dgr"])
        plt.show()


# TODO: Remove
if __name__ == "__main__":
    # Delete the directory
    logging.basicConfig(level=logging.INFO)
    samples_per_param = 100
    num_conditions = 1
    cache = True
    # shutil.rmtree(
    #     Path(f'/home/jason/Documents/Uni/thesis/gtfa/results/toy_likelihood_{num_conditions}_{samples_per_param}_all'),
    #     ignore_errors=True)
    sample_and_show(Path("/home/jason/Documents/Uni/thesis/gtfa/model_configurations/toy_likelihood_conc_single.toml"),
                    samples_per_param=samples_per_param, num_conditions=num_conditions, cache=cache)
