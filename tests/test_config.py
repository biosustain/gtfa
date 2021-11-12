
# name: str
# stan_file: Path
# data_folder: Path
# sample_kwargs: Dict
# analyse: dict
# likelihood: bool
# order: list = None
# result_dir: Path = Path("empty")
# devel: bool = True
# disp_plot: bool = True
# save_plot: bool = False
# verbose: bool = True
from pathlib import Path

from src.model_configuration import ModelConfiguration


def test_config_defaults():
    mc = ModelConfiguration(
        name="test",
        stan_file=Path(""),
        data_folder=Path(""),
        sample_kwargs={},
        analyse={"test": "test"},
        likelihood=True
    )
    # Test the defaults
    assert mc.order is None
    assert mc.result_dir == Path("empty")
    assert mc.devel is True
    assert mc.disp_plot is True
    assert mc.save_plot is False
    assert mc.verbose is True


def test_config_defaults_none_input():
    """ Default values should override none inputs"""
    mc = ModelConfiguration(
        name="test",
        stan_file=Path(""),
        data_folder=Path(""),
        sample_kwargs={},
        analyse={"test": "test"},
        likelihood=True,
        order=None,
        result_dir=None,
        devel=None,
        disp_plot=None,
        save_plot=None,
        verbose=None
    )
    # Test the defaults
    assert mc.order is None
    assert mc.result_dir == Path("empty")
    assert mc.devel is True
    assert mc.disp_plot is True
    assert mc.save_plot is False
    assert mc.verbose is True