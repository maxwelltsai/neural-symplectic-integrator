# Sam Greydanus, Misko Dzama, Jason Yosinski
# 2019 | Google AI Residency Project "Hamiltonian Neural Networks"

import numpy as np
import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


def integrate_model(model, t_span, y0, fun=None, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  print('Integrating..., y0=', y0)
  fun = default_fun if fun is None else fun
  return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def rk4(fun, y0, t, dt, *args, **kwargs):
  dt2 = dt / 2.0
  k1 = fun(y0, t, *args, **kwargs)
  k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
  k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
  k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
  dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
  return dy


def L2_loss(u, v):
  return (u-v).pow(2).mean()


def custom_loss(dxdt, dxdt_hat, x):
    l2_loss = L2_loss(dxdt, dxdt_hat)
    dLdt = torch.cross(dxdt_hat[:,3:], x[:, 3:]) - torch.cross(x[:,3:], dxdt_hat[:, :3])
    magnitude_dLdt = torch.linalg.norm(dLdt.sum(axis=0))
    # print(dLdt)
    scaling_factor = 1
#     print(scaling_factor, torch.exp(scaling_factor))
    return scaling_factor * l2_loss + magnitude_dLdt * 100


def tanh_log(x):
  return torch.tanh(x) * torch.log(torch.tanh(x) * x + 1)

  # threshold = 2.69364
  # y0 = torch.tanh(x[torch.abs(x) <= threshold])
  # y1 = torch.log(x[x > threshold])
  # y2 = -torch.log(-x[x < -threshold])
  # print(y2.shape, y1.shape, y0.shape, torch.abs(x).shape)
  # y = torch.cat((y2, y0, y1), 0)
  # print(x.shape, y.shape, y2.shape, y1.shape, y0.shape)
  # y[x >= threshold] = torch.log(x[x >= threshold])
  # y[x <= -threshold] = -torch.log(-x[x <= -threshold])
    
  return y

class TanhLog(torch.nn.Module):
  '''
  Applies the Sigmoid Linear Unit (SiLU) function element-wise:
      SiLU(x) = x * sigmoid(x)
  Shape:
      - Input: (N, *) where * means, any number of additional
        dimensions
      - Output: (N, *), same shape as the input
  References:
      -  Related paper:
      https://arxiv.org/pdf/1606.08415.pdf
  Examples:
      >>> m = silu()
      >>> input = torch.randn(2)
      >>> output = m(input)
  '''
  def __init__(self):
    '''
    Init method.
    '''
    super().__init__() # init the base class

  def forward(self, x):
      '''
      Forward pass of the function.
      '''
      threshold = 2.69364
      y0 = torch.tanh(x[torch.abs(x) <= threshold])
      y1 = torch.log(x[x > threshold])
      y2 = -torch.log(-x[x < -threshold])
      print(y2.shape, y1.shape, y0.shape, torch.abs(x).shape)
      y = torch.cat((y2, y0, y1), 0)
      print(x.shape, y.shape, y2.shape, y1.shape, y0.shape)
      return y # simply apply already implemented SiLU

def read_lipson(experiment_name, save_dir):
  desired_file = experiment_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)


def str2array(string):
  lines = string.split('\\n')
  names = lines[0].strip("b'% \\r").split(' ')
  dnames = ['d' + n for n in names]
  names = ['trial', 't'] + names + dnames
  data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]

  return np.asarray(data), names


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  elif name == 'lrelu':
    nl = torch.nn.LeakyReLU(0.1)
  elif name == 'tanh_log':
    nl = tanh_log
  else:
    raise ValueError("nonlinearity not recognized")
  return nl


def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
    '''Given a three dimensional array [frames, height, width], make
    a gif and save it.'''
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    for i in range(len(frames)):
        im = (frames[i].clip(-.5,.5) + .5)*255
        im[divider,:] = 0
        im[divider + 1,:] = 255
        if pixels is not None:
          im = scipy.misc.imresize(im, pixels)
        scipy.misc.imsave(temp_dir + '/f_{:04d}.png'.format(i), im)

    images = []
    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(temp_dir, file_name)
            images.append(imageio.imread(file_path))
    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)

    shutil.rmtree(temp_dir) # remove all the images
    return png_save_path
