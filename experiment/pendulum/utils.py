import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from collections import OrderedDict

from datasets.oscillations import create_Oscillation

from ruamel.yaml import YAML
from pathlib import Path, PosixPath

class MyDataset(Dataset):
  def __init__(self,d, seq_len):
    self.xm = d[:,0:2] #größe data
    self.seq_len = seq_len # batch#
    self.xa = d[:,2:]


  def __len__(self):
    return self.xm.shape[0]-self.seq_len+1

  def __getitem__(self, i):
    return (self.xm[i:i+self.seq_len],self.xa[i:i+self.seq_len])


class MyLSTMDataset(Dataset):
  def __init__(self,d,seq_len):
    self.d = d # 250
    self.seq_len = seq_len #25 batch

  def __len__(self):
    return self.d.shape[0]-self.seq_len+1

  def __getitem__(self, i):
    return (self.d[i:i+self.seq_len], self.d[i+self.seq_len:i+self.seq_len+1])


def normalize(train):
    scaler = MinMaxScaler(feature_range=(0,1))
    train_norm = scaler.fit_transform(train.reshape(-1, 2))
    train_norm = torch.FloatTensor(train_norm)
    return scaler, train_norm


def create_batch(input_data, tw):
  bat = []
  L = input_data.shape[0]
  for i in range(L-tw):
    train_seq = input_data[i:i+tw]
    train_label = input_data[i+tw:i+tw+1]
    bat.append((train_seq ,train_label))
  return bat

def plot_training(out_file, t, E_kin, E_pot, pred_E_kin, pred_E_pod, friction, seq_len, title_appendix):

  plt.figure(figsize=(8.0, 6.0))
  if friction:
    plt.title('Pendulum with friction' + title_appendix)
  else:
    plt.title('Pendulum without friction' + title_appendix)


  plt.ylabel('Energy in J')
  plt.autoscale(axis='t', tight=True)
  plt.plot(t, E_kin, label='Kinetic Energy', c = "black")
  plt.plot(t, E_pot, label='Potential Energy', c = "#d3d3d3")
  plt.plot(t[:seq_len], pred_E_kin, label ='Predicted Kinetic Energy', c="magenta", linewidth=3)
  plt.plot(t[:seq_len], pred_E_pod, label ='Predicted Potential Energy', c="cyan", linewidth=3)
  plt.legend()

  plt.rcParams.update({'font.size': 12})
  plt.savefig(out_file, bbox_inches='tight')
  plt.close('all')
  plt.clf()


def plot_test(out_file,
              t,
              E_kin,
              E_pot,
              pred_E_kin,
              pred_E_pod,
              friction,
              length,
              modeltype: str= "MC-LSTM"):
  plt.close('all')
  plt.figure(figsize=(7.5, 6.0))

  if friction:
    plt.title(modeltype + ': Pendulum with friction')
  else:
    plt.title(modeltype + ': pendulum without friction')

  plt.ylabel('Energy in J')
  plt.xlabel('Time')
  plt.autoscale(axis='t', tight=True)
  plt.plot(t, E_kin, label='Kinetic Energy (KE)', c = "black")
  plt.plot(t, E_pot, label='Potential Energy (PE)', c = "#d3d3d3")

  plt.plot(t[:length], pred_E_kin[:length], label ='Train-Predictions KE', c="magenta", linewidth=3)
  plt.plot(t[:length], pred_E_pod[:length], label ='Train-Predictions PE', c="cyan", linewidth=3)

  plt.plot(t[length:], pred_E_kin[length:], label ='Test-Predictions KE', c="red", linewidth=3)
  plt.plot(t[length:], pred_E_pod[length:], label ='Test-Predictions PE', c="dodgerblue", linewidth=3)

  plt.axvline(t[length-1], -1, 1, c="black")
  plt.legend()

  plt.rcParams.update({'font.size': 12})
  plt.savefig(out_file, bbox_inches='tight', dpi = 300)

  plt.clf()
  plt.close('all')


def get_split(test_len,
              norm,
              lam,
              friction,
              pendulum_length = 1,
              initial_amplitude = 0.3,
              noise_std = 0.01
              ):
  #get data
  df = create_Oscillation(lam,
                          friction,
                          pendulum_length = pendulum_length,
                          initial_amplitude = initial_amplitude)

  noise = np.random.normal(0, noise_std, size=df.shape)
  df += noise
  #
  d = df[['Kinetic Energy', 'Potential Energy']].values.astype(float)
  #aux_primary = df[['Angle', 'Deflection']].values.astype(float)
  time_line = np.linspace(0, np.pi, df.shape[0])
  aux = np.stack([
    np.sin(10 * time_line),
    np.sin(20 * time_line),
    np.sin(30 * time_line),
    np.sin(40 * time_line),
    np.sin(50 * time_line),
    np.sin(100 * time_line),
    np.sin(200 * time_line),
    np.sin(400 * time_line),
    np.sin(600 * time_line)
  ], axis=1)
  #aux = np.concatenate([aux_primary, time_marker], axis = 1)

  train = d[:-test_len]
  train_aux = aux[:-test_len]
  train_aux = torch.FloatTensor(train_aux) #.view(-1,1)
  #test = d[-test_len:]
  #test_aux = aux[-test_len:].reshape(-1, 2)
  #test_aux = torch.FloatTensor(test_aux)#.view(-1,1)
  test = d
  test_aux = torch.FloatTensor(aux)

  if norm:
    scaler = MinMaxScaler(feature_range=(0,1))
    train_norm = scaler.fit_transform(train.reshape(-1, 2))
    train = torch.FloatTensor(train_norm)
    test_norm = scaler.transform(test)
    test = torch.FloatTensor(test_norm)
  else:
    test = torch.FloatTensor(test)
    train = torch.FloatTensor(train)
    scaler = None

  return df, train, test, scaler, train_aux, test_aux

def read_pendulum_config(cfg_path):
    if cfg_path.exists():
      with cfg_path.open('r') as fp:
        yaml = YAML(typ="safe")
        cfg = yaml.load(fp)
    else:
      raise FileNotFoundError(cfg_path)
    return cfg


def dump_pendulum_config(cfg, folder, filename: str = 'config.yml'):
  cfg_path = folder / filename
  if not cfg_path.exists():
    with cfg_path.open('w') as fp:
      temp_cfg = {}
      for key, val in cfg.items():
        if any([x in key for x in ['dir', 'path', 'file']]):
          if isinstance(val, list):
            temp_list = []
            for elem in val:
              temp_list.append(str(elem))
            temp_cfg[key] = temp_list
          else:
            temp_cfg[key] = str(val)
        else:
          temp_cfg[key] = val
      yaml = YAML()
      yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
  else:
    FileExistsError(cfg_path)