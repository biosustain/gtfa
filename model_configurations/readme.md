This folder contains [toml](https://github.com/toml-lang/toml) files that represent model/configuration combinations that you might want to run.

See
[here](https://github.com/biosustain/gtfa/blob/master/src/model_configuration.py)
for the python representation of a model configuration, i.e. a
`ModelConfiguration` dataclass.

The fields are as follows:

- `name`: a non-empty string
- `stan_file`: a path to a `.stan` file, starting at the project root
- `data_folder`: a path to a folder containing data, starting at the project root
- `likelihood`: a boolean indicating whether or not to take into account
  measurements
- `sample_kwargs` a subtable of valid inputs to the [`cmdstanpy`](https://cmdstanpy.readthedocs.io) method `CmdStanModel.sample`

Note that, unlike in Python, booleans in toml are lower case.
