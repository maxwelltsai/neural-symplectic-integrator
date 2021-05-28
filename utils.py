
import numpy as np
import os, torch, pickle, zipfile
from config import CONFIG


def L2_loss(u, v):
  return (u-v).pow(2).mean()


def custom_loss(dxdt, dxdt_hat, x):
    l2_loss = L2_loss(dxdt, dxdt_hat)
    dLdt = torch.cross(dxdt_hat[:,3:], x[:, 3:]) - torch.cross(x[:,3:], dxdt_hat[:, :3])
    magnitude_dLdt = torch.linalg.norm(dLdt.sum(axis=0))
    scaling_factor = 1
    return scaling_factor * l2_loss + magnitude_dLdt * 100


def tanh_log(x):
  return torch.tanh(x) * torch.log(torch.tanh(x) * x + 1)
    
def get_model_path():
  config = CONFIG
  path = '{}/model_{}_{}.pth'.format(config['save_dir'], config['backbone'], config['activation'])
  return path

def get_backbone(name):
  config = CONFIG
  import nn_models 
  return getattr(nn_models, name)(config['input_dim'], config['hidden_dim'], config['output_dim'], config['activation'])

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def choose_activation(name):
  act = None
  if name == 'tanh':
    act = torch.tanh
  elif name == 'relu':
    act = torch.relu
  elif name == 'sigmoid':
    act = torch.sigmoid
  elif name == 'softplus':
    act = torch.nn.functional.softplus
  elif name == 'selu':
    act = torch.nn.functional.selu
  elif name == 'elu':
    act = torch.nn.functional.elu
  elif name == 'swish':
    act = lambda x: x * torch.sigmoid(x)
  elif name == 'lrelu':
    act = torch.nn.LeakyReLU(0.1)
  elif name == 'SymmetricLog':
    act = tanh_log
  else:
    raise ValueError("Activation function not recognized")
  return act


