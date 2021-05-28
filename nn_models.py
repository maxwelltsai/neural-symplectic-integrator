# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np
from utils import choose_nonlinearity
import os.path

import scipy.sparse as sparse

class MLP_dropout(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.Dropout1d = torch.nn.Dropout(0.5)
    self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)
      
  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    h = self.Dropout1d(h)
    return self.linear5(h)

class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear8]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    if not os.path.isfile('mask_file'):
      mask = torch.tensor(sparse.random(hidden_dim, hidden_dim, density=0.1).toarray())
      torch.save(mask, "mask_file")
    else:
      mask = torch.load("mask_file")

    with torch.no_grad():
      self.linear2.weight.mul_(mask)
      self.linear3.weight.mul_(mask)
      self.linear4.weight.mul_(mask)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    return self.linear8(h)

class MLP_vanilla(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    return self.linear5(h)

class MLPAutoencoder(torch.nn.Module):
  '''A salt-of-the-earth MLP Autoencoder + some edgy res connections'''
  def __init__(self, input_dim, hidden_dim, latent_dim, nonlinearity='tanh'):
    super(MLPAutoencoder, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

    self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
    self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, \
              self.linear5, self.linear6, self.linear7, self.linear8]:
      torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def encode(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = h + self.nonlinearity( self.linear2(h) )
    h = h + self.nonlinearity( self.linear3(h) )
    return self.linear4(h)

  def decode(self, z):
    h = self.nonlinearity( self.linear5(z) )
    h = h + self.nonlinearity( self.linear6(h) )
    h = h + self.nonlinearity( self.linear7(h) )
    return self.linear8(h)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat