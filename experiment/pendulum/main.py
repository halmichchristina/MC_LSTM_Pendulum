"""
Pendulum experiment main function
"""

import torch
import time
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from experiment.pendulum.utils import plot_training, plot_test, get_split, MyDataset
from models.model import NoInputMassConserving, JustAnARLSTM

def run_pendulum_experiment(cfg):
    # Define model type:
    modeltype = cfg['modeltype']

    # Define parameters for oscillation
    # Damping constant
    lam = cfg['dampening_constant']
    if lam == 0:
        friction = False
    else:
        friction = True

    # Define parameters for training
    seq_len = np.min([ 980, cfg['train_seq_length'] ])
    test_len = 1000 - seq_len
    pendulum_length = cfg['pendulum_length']
    initial_amplitude = cfg['initial_amplitude']
    noise_std = cfg['noise_std']

    # --------------------------------------------------------------------------
    # fixed settings found by student-descent:
    batch = 1
    lr = 0.01
    if (not friction):
        epochs = 1500
    else:
        epochs = 1500
    scale = True
    norm = True  # True

    # --------------------------------------------------------------------------
    # plotting and saving information:
    if friction:
        friction_addendum = "friction"
    else:
        friction_addendum = "non_friction"

    # --------------------------------------------------------------------------
    # Get data, create loader
    df, train, test, scaler, train_aux, test_aux = get_split(
                    test_len,
                    norm,
                    lam,
                    friction,
                    pendulum_length = pendulum_length,
                    initial_amplitude = initial_amplitude,
                    noise_std=noise_std)
    train_cat = torch.cat((train, train_aux), dim=1)
    ds = MyDataset(train_cat, seq_len)

    loader = DataLoader(ds, shuffle=True, batch_size=batch)

    # --------------------------------------------------------------------------
    # model and optimizer
    if modeltype == "MC-LSTM":
        model = NoInputMassConserving(hidden_size=2,
                                      initial_output_bias=-5,
                                      scale_c=scale,
                                      hidden_layer_size=100,
                                      friction=friction)
    elif modeltype == "AR-LSTM":
        model = JustAnARLSTM()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    base_loss = nn.MSELoss()

    ############################################################################
    # Training
    start_train = time.time()
    model.train()
    plot_idx = 0
    tau = 11 # how many timesteps are used to compute loss
    for i in range(epochs):
        for b, (xm, xa) in enumerate(loader):
            optimizer.zero_grad()

            m_out, c = model(xm[:, 0],
                             xm.shape[1] - 1,
                             xa=xa)

            # compute correlations:
            vx = xm[:, 1:tau, 0] - torch.mean(xm[:, 1:tau, 0])
            vy = c[:, 1:tau, 0] - torch.mean(c[:, 1:tau, 0])
            cor0 = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            vx = xm[:, 1:tau, 1] - torch.mean(xm[:, 1:tau, 1])
            vy = c[:, 1:tau, 1] - torch.mean(c[:, 1:tau, 1])
            cor1 = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
            # compute overall loss:
            single_loss = base_loss(c[:, 1:tau, :], xm[:, 1:tau, :]) - (cor0 + cor1) / 2

            # optimize:
            single_loss.backward()
            optimizer.step()

            # advance curriculum:
            if single_loss.item() <= -0.9:  # fixed, why not
                tau = np.min([tau + 5, xm.shape[1] + 1])

        if cfg["create_plots"] & (i % 100 == 1):
            print(f'epoch: {i} loss: {single_loss.item():10.8f}')

            with torch.no_grad():
                m_out, c = model(xm[:, 0], seq_len - 1, xa=xa)
            if norm:
                act = scaler.inverse_transform(c.squeeze(0).numpy())
            else:
                act = c.squeeze(0).numpy()

            plot_idx += 1
            name = f"plot_{modeltype}_{friction_addendum}_idx{plot_idx:03d}.png"
            plot_training(Path(cfg["out_dir"], "figures", name),
                          df['Time'],
                          df['Kinetic Energy'],
                          df['Potential Energy'],
                          act[:, 0],
                          act[:, 1],
                          friction,
                          seq_len,
                          title_appendix=f'(epoch {i})')

    # --------------------------------------------------------------------------
    # save params
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("training finished")
    if modeltype == "MC-LSTM":
        torch.save(model.state_dict(),
                   Path(cfg["out_dir"], f'mc-lstm_seed{cfg["seed"]}__{friction_addendum}_params.p'))
    elif modeltype == "AR-LSTM":
        torch.save(model.state_dict(),
                   Path(cfg["out_dir"], f'ar-lstm_seed{cfg["seed"]}__{friction_addendum}_params.p'))

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    print('Time training ', (time.time() - start_train) / 60)


    model.eval()
    with torch.no_grad():
        m_out, c = model(test[0].unsqueeze(0),
                         seq_len + test_len - 1,
                         xa=test_aux.unsqueeze(0))
        if norm:
            actual_predictions = scaler.inverse_transform(c.squeeze(0).numpy())
        else:
            actual_predictions = c.squeeze(0).numpy()


    if cfg["create_plots"]:
        name = f"plot_{modeltype}_{friction_addendum}__{cfg['experiment_name']}.png"
        plot_test(Path(cfg["out_dir"], "figures_test", name),
                  df['Time'],
                  df['Kinetic Energy'],
                  df['Potential Energy'],
                  actual_predictions[:, 0],
                  actual_predictions[:, 1],
                  friction,
                  seq_len,
                  modeltype)

    # fix actual test with 500 timesteps to have equal time-steps through experiments
    test_data = pd.DataFrame({
        "obs_kin":df['Kinetic Energy'][-test_len:(-test_len+500)], # use test_len to avoiud + 1, fix mse for 500 timesteps
        "obs_pot":df['Potential Energy'][-test_len:(-test_len+500)],
        "sim_kin":actual_predictions[-test_len:(-test_len+500),0],
        "sim_pot":actual_predictions[-test_len:(-test_len+500),0] })

    # a bit ugly
    test_mse = 0.5*np.mean((test_data["obs_kin"] - test_data["sim_kin"])**2)    # mse kinetic energy
    test_mse += 0.5*np.mean((test_data["obs_pot"] - test_data["sim_pot"])**2)   # mse potential energy

    print(test_mse)
    return test_data, test_mse



