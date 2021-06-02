"""Define a list of ModelConfiguration objects called MODEL_CONFIGURATIONS."""

import os
from .util import get_99_pct_params_ln, get_99_pct_params_n
from .model_configuration import ModelConfiguration


# Location of this file
HERE = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(HERE, "..", "data", "raw")

# Configure cmdstanpy.CmdStanModel.sample
SAMPLE_KWARGS = dict(
    show_progress=True,
    save_warmup=False,
    fixed_param=True,
    iter_sampling=1,
    inits=0.01,
)

# Configuration of model.stan with no A:B interaction
TOY_PRIOR = ModelConfiguration(
    name="toy_prior",
    stan_file=os.path.join(HERE, "stan", "model.stan"),
    data_folder=os.path.join(RAW_DATA_DIR, "toy_model"),
    sample_kwargs=SAMPLE_KWARGS,
    likelihood=False
)
TOY_LIKELIHOOD = ModelConfiguration(
    name="toy_likelihood",
    stan_file=os.path.join(HERE, "stan", "model.stan"),
    data_folder=os.path.join(RAW_DATA_DIR, "toy_model"),
    sample_kwargs=SAMPLE_KWARGS,
    likelihood=False
)

# A list of model configurations to test
MODEL_CONFIGURATIONS = [TOY_PRIOR, TOY_LIKELIHOOD]
