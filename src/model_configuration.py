"""Definition of the ModelConfiguration class."""
import dataclasses
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Any
import pandas as pd
import toml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    """Container for a path to a Stan model and some configuration.

    For example, you may want to compare how well two stan programs fit the
    same data, or how well the same model fits the data with different
    covariates.

    :param name: string name identifying the model configuration

    :param stan_file: Path to a Stan program

    :param data_folder: Path to a folder containing files "measurements.csv", "stoichiometry.csv" and "priors.csv"

    :param result_dir: Path to the result files of this run

    :param sample_kwargs: dictionary of keyword arguments to
    cmdstanpy.CmdStanModel.sample.
    
    :param likelihood: take measurements into account

    :param devel: This is being run for development and can overwrite previous values

    :param run: run this config with the script `fit_all_model_configurations.py`

    :param analyse: run analysis functions and save results
    """

    name: str
    stan_file: Path
    data_folder: Path
    sample_kwargs: Dict
    analyse: dict
    likelihood: bool
    order: list = None
    result_dir: Path = Path("empty")
    devel: bool = True
    disp_plot: bool = True
    save_plot: bool = False
    verbose: bool = True

    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, dataclasses._MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

    def __copy__(self):
        """Return a copy of the object."""
        return dataclasses.replace(self)

    def copy(self):
        """Return a copy of the object."""
        return self.__copy__()

    def to_toml(self, path: Path):
        """Write a toml string representation of the object."""
        with open(path, "w") as f:
            dict = dataclasses.asdict(self)
            for k, v in dict.items():
                if isinstance(v, Path):
                    dict[k] = str(v)
            toml.dump(dict, f)


def load_model_configuration(path: Path) -> ModelConfiguration:
    d = toml.load(path)
    # Warn if there are extra fields in the TOML that aren't expected
    extra = d.keys() - ModelConfiguration.__annotations__.keys()
    if extra:
        extra_str = ', '.join(map(str, extra))
        logger.warning(f"The following unexpected params in the config file were ignored: {extra_str}")
    mc = ModelConfiguration(
        name=d.get("name"),
        stan_file=Path(d.get("stan_file")),
        data_folder=Path(d.get("data_folder")),
        likelihood=d.get("likelihood"),
        sample_kwargs=d.get("sample_kwargs"),
        analyse=d.get("analyse"),
        devel=d.get("devel"),
        verbose=d.get("verbose"),
        order=d.get("order")
    )
    # Paths are relative to the config file
    if not mc.result_dir.is_absolute():
        mc.result_dir = Path(path).parent / mc.result_dir
    if not mc.data_folder.is_absolute():
        mc.data_folder = Path(path).parent / mc.data_folder
    if not mc.stan_file.is_absolute():
        mc.stan_file = Path(path).parent / mc.stan_file
    # Check the paths
    validate_model_configuration(mc)
    return mc


def validate_model_configuration(mc: ModelConfiguration) -> None:
    assert os.path.exists(mc.stan_file), f"stan file {mc.stan_file} does not exist"
    assert os.path.exists(mc.data_folder), f"data folder {mc.data_folder} does not exist"
    assert type(mc.name) is str, "name must be a string"
    assert mc.name != ""
    assert type(mc.likelihood) is bool
