"""Define a list of ModelConfiguration objects called MODEL_CONFIGURATIONS."""

import os

from .model_configuration import ModelConfiguration
from .pandas_to_cmdstanpy import SAMPLE_KWARGS


# Location of this file
HERE = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(HERE, "..", "data", "raw")

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
    likelihood=True
)

# A list of model configurations to test
MODEL_CONFIGURATIONS = [TOY_LIKELIHOOD, TOY_PRIOR]
