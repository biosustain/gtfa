import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from matplotlib import pyplot as plt


from src.cmdstanpy_to_arviz import get_infd_kwargs
from src.fitting import stan_input_from_dir
from src.model_configuration import load_model_configuration

import arviz as az



def run_standard_model():
    """
    Assumes that the data follows exactly the model as described in the concentration model.
    """
    test_dir = Path("test_dir")
    result_dir = test_dir / "results"
    S = pd.read_csv(test_dir / "stoichiometry.csv", index_col=0)
    # These were determined by samples that seem somewhat reasonable
    c = np.array([-7.97931242, -6.76317324, -8.5203623,  -8.34846125])
    b = np.array([54.479056089260254, 92.27813827634556, 2940.255152937493, 13.922257487919769, 60.519135042731186])
    v = np.array([1.0, -1.0, -0.9275499473602622, -0.7989785015503799, 0.2010214984495633, 0.07245005263970583, 0.12857144580990365])
    e = np.array([0.003641157529874064, 0.0023508286235591246, 0.0004050401475330379, 0.0006353773420867625, 0.0006045197331044093])
    write_measurements(test_dir, S, c, v, e, err=1)
    # Now run the sampling
    config = load_model_configuration("/home/jason/Documents/Uni/thesis/gtfa/src/tests/test_small_likelihood.toml")
    config.result_dir = result_dir
    # Run the sampling
    model = CmdStanModel(
        model_name=config.name, stan_file=str(config.stan_file)
    )
    stan_input = stan_input_from_dir(test_dir, likelihood=config.likelihood)
    # Make the samples directory
    sample_dir = config.result_dir / "samples"
    sample_dir.mkdir()
    mcmc = model.sample(
        data=stan_input,
        output_dir=str(sample_dir),
        save_warmup=True,
        **config.sample_kwargs,
    )
    print(mcmc.diagnose().replace("\n\n", "\n"))
    # TODO: This should be cleaner

    # TODO: Just run the standard one normally, not here
    # TODO: Continue here. ValueError: conflicting sizes for dimension 'condition': length 1 on the data but length 2 on coordinate 'condition'
    measurements = pd.read_csv(config.data_folder / "measurements.csv")
    priors = pd.read_csv(config.data_folder / "priors.csv")
    infd_kwargs = get_infd_kwargs(S, measurements, priors, config.order, config.sample_kwargs)
    infd = az.from_cmdstan(
        mcmc.runset.csv_files, **infd_kwargs
    )
    infd.to_netcdf(str(config.result_dir / "infd.nc"))
    idata = az.from_netcdf(config.result_dir / "infd.nc")
    az.plot_density(idata, var_names=["log_metabolite", "b", "flux", "enzyme"])
    az.plot_pair(idata, var_names=["dgr"])
    plt.show()


def write_measurements(test_dir, S, c, v, e, err=1.0):
    num_mets = len(c)
    num_rxns = len(v)
    num_internal = len(e)
    # Write this to the data file
    cols = ["measurement_type", "target_id", "condition_id", "measurement", "error_scale"]
    measurements = pd.DataFrame(columns=cols)
    # Metabolite concentrations
    # TODO: REVERT
    conc_df = pd.DataFrame(zip(["mic"] * num_mets, S.index, ["condition_1"] * num_mets, np.exp(c), num_mets * [0.2 * err]),
                           columns=cols)
    measurements = measurements.append(conc_df)
    # Fluxes
    flux_df = pd.DataFrame(zip(["flux"] * num_rxns, S.columns, ["condition_1"] * num_rxns, v, [0.2 * err] * num_mets),
                           columns=cols)
    measurements = measurements.append(flux_df)
    # Enzyme concentrations
    enz_df = pd.DataFrame(
        zip(["enzyme"] * num_internal, S.columns, ["condition_1"] * num_internal, e, [0.2 * err] * num_internal),
        columns=cols)
    measurements = measurements.append(enz_df)
    measurements.to_csv(test_dir / "measurements.csv", index=False)


test_dir = Path("test_dir")
result_dir = test_dir / "results"


# Here the models are written but it's slow
if test_dir.exists():
    shutil.rmtree(test_dir)
test_dir.mkdir()
result_dir.mkdir()
from src import model_conversion
from src.tests.model_setup import build_small_test_model
tmodel = build_small_test_model()
model_conversion.write_files_from_tmodel(tmodel, test_dir)


# If you're not writing the model use this to clear the directory
# if result_dir.exists():
#     shutil.rmtree(result_dir)
# result_dir.mkdir()

run_standard_model()