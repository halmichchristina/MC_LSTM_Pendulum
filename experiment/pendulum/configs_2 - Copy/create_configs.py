import itertools

from pathlib import Path
import os

from experiment.pendulum.utils import read_pendulum_config, dump_pendulum_config


# ------------------------------------------------------------------------------
#cfg_path = "../config.yml"
cfg_path = os.getcwd() + "\\experiment\\config.yml"
#out_dir = "./"
out_dir = os.getcwd() + "\\experiment\\pendulum\\configs_2\\"

# ------------------------------------------------------------------------------
def iterate_values(S):
    keys, values = zip(*S.items())

    for row in itertools.product(*values):
        yield dict(zip(keys, row))


# ------------------------------------------------------------------------------
hyper_parameters = {
    "modeltype": ["MC-LSTM","AR-LSTM"],
    "dampening_constant": [0.0],#, 0.1, 0.2, 0.4, 0.8],
    "initial_amplitude": [0.2, 0.3, 0.4],
     "pendulum_length": [0.75, 1],
    "train_seq_length": [100, 200, 400],
    "noise_std": [0.0, 0.01] # in current setup: max = 499
 }

param_list = list(iterate_values(hyper_parameters))

# ------------------------------------------------------------------------------
for idx, param in enumerate(param_list):
    cfg_mod = read_pendulum_config(Path(cfg_path))
    experiment_name = cfg_mod["experiment_name"] + f"_{idx + 1}"
    for key, value in param.items():
        cfg_mod[key] = value
    cfg_mod["experiment_name"] = experiment_name
    dump_pendulum_config(cfg_mod, Path(out_dir), f'config_{idx + 1}.yml')
