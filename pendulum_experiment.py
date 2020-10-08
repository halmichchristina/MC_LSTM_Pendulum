import argparse
import torch
import random
import glob
#import feather

import pandas as pd
import numpy as np

from pathlib import Path
from experiment.pendulum.utils import read_pendulum_config
from experiment.pendulum import run_pendulum_experiment


# ------------------------------------------------------------------------------
# get basic arguments from config-file:
parser = argparse.ArgumentParser()
parser.add_argument('--config-dir', type=str)
args = vars(parser.parse_args())

# ------------------------------------------------------------------------------
# read in config-files:
config_files = glob.glob(args["config_dir"] + '*.yml')

mse_experiments = []
idx = 0
for config_file in config_files:
    idx += 1
    cfg = read_pendulum_config(Path(config_file))

    # ------------------------------------------------------------------------------
    # set seeds:
    if cfg["seed"] is None:
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
    # fix random seeds for various packages
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ------------------------------------------------------------------------------
    # conduct experiment:
    test_data, final_mse = run_pendulum_experiment(cfg)
    mse_experiments = {"modeltype":cfg["modeltype"],
                       "dampening_constant":cfg["dampening_constant"],
                       "train_seq_length": cfg["train_seq_length"],
                       "noise_std":cfg["noise_std"],
                        "initial_amplitude":cfg["initial_amplitude"],
                        "pendulum_length":cfg["pendulum_length"],
                        "mse":final_mse}
    mse_experiments = pd.DataFrame(mse_experiments, index=[0])
    #feather.write_dataframe(mse_experiments, Path(cfg["out_dir"], "data", f'mse_{idx}_{cfg["experiment_name"]}.f'))
    #feather.write_dataframe(test_data, Path(cfg["out_dir"], "data", f'data_{idx}_{cfg["experiment_name"]}.f'))

#results = pd.DataFrame(mse_experiments)
#feather.write_dataframe(results, Path(cfg["out_dir"], f'results_{cfg["out_appendix"]}.f'))
print("wuhu")

