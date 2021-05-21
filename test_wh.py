from nn_models import MLP
from hnn import HNN
import torch
import sys
from wh import WisdomHolman
import numpy as np
import copy
import matplotlib.pyplot as plt


DPI = 300
FORMAT = 'pdf'
EXPERIMENT_DIR = '.'
sys.path.append(EXPERIMENT_DIR)

def get_args():
    return {'input_dim': 6, # two bodies, each with q_x, q_y, p_z, p_y
         'hidden_dim': 512,
         'learn_rate': 1e-4,
         'input_noise': 0.,
         'batch_size': 200,
         'nonlinearity': 'tanh',
         'total_steps': 10000,
         'field_type': 'solenoidal',
         'print_every': 200,
         'verbose': True,
         'name': 'wh',
         'seed': 3,
         'save_dir': '{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

def load_model(args, baseline=False):
    output_dim = args.input_dim if baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    print(nn_model)
    model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=baseline)

    case = 'baseline' if baseline else 'hnn'
    path = "{}/{}-orbits-{}.tar".format(args.save_dir, args.name, case)
    print(path)
    model.load_state_dict(torch.load(path))
    return model

args = ObjectView(get_args())
print(args.field_type, args.input_dim)
# base_model = load_model(args, baseline=True).cuda()
hnn_model = load_model(args, baseline=False).cuda()

# initial conditions
semi = 0.5
a0 = semi
# a1 = semi + np.random.rand() + 1
# a1 = semi * 1 + 5 * np.random.rand() + 1
a1 = 0.7
t_end = 500
h = 0.2

wh_nih = WisdomHolman(hnn=hnn_model)
wh_nih.particles.add(mass=1.0, pos=[0., 0., 0.,], vel=[0., 0., 0.,], name='Sun')

for i in range(100):
    wh_nih.particles.add(mass=1.e-8, a=np.random.rand() + 0.5, e=0.05*np.random.rand(), i=0, name='planet%d' % i, primary='Sun',f=2*np.pi*np.random.rand())


# wh_nih.particles.add(mass=1.e-4, a=a0, e=0.05*np.random.rand(), i=np.pi/2*np.random.rand(), name='planet1', primary='Sun',f=2*np.pi*np.random.rand())
# wh_nih.particles.add(mass=1.e-4, a=a0, e=0.05*np.random.rand(), i=0, name='planet1', primary='Sun',f=2*np.pi*np.random.rand())
# wh_nih.particles.add(mass=1.e-6, a=a1, e=0.1*np.random.rand(), i=np.pi/2*np.random.rand(), name='planet2', primary='Sun',f=2*np.pi*np.random.rand())
# wh_nih.particles.add(mass=1.e-4, a=a0, e=0.05, i=np.pi/30, name='planet1', primary='Sun',f=2*np.pi*np.random.rand())
# wh_nih.particles.add(mass=1.e-4, a=a1, e=0.03, name='planet2', primary='Sun',f=2*np.pi*np.random.rand())
# wh_nih.particles.add(mass=1.e-4, a=2*a1, e=0.03, name='planet2', primary='Sun',f=2*np.pi*np.random.rand())
wh_nih.output_file = 'data.h5'
wh_nih.store_dt = 10


# save an initial state of the particle sets to compare with other methods (e.g., traditional N-body method)
particles_init = copy.deepcopy(wh_nih.particles)

# initialize the integrator
wh_nih.integrator_warmup()
wh_nih.h = h
wh_nih.acceleration_method = 'numpy'
wh_nih.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
wh_nih.buf.recorder.start()

# start to propagate
wh_nih.integrate(t_end, nih=True)
wh_nih.buf.flush()

# save the data
energy_nih = wh_nih.energy
data_nih = wh_nih.buf.recorder.data
wh_nih.stop()

# create a traditional WH integrator as the ground truth generator
wh_nb = WisdomHolman(particles=particles_init)
wh_nb.integrator_warmup()
wh_nb.h = h
wh_nb.acceleration_method = 'numpy'
wh_nb.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z'])
wh_nb.buf.recorder.start()
wh_nb.integrate(t_end, nih=False)
wh_nb.buf.flush()
energy_nb = wh_nb.energy
data_nb = wh_nb.buf.recorder.data

plt.figure(0)
plt.plot(energy_nih, label='wh-nih')
plt.plot(energy_nb, label='wh')
plt.xlabel('$t$ [yr]')
plt.ylabel('dE/E0')
plt.yscale('log')
plt.legend()
plt.savefig('compare_energy.pdf')
plt.close(0)

plt.figure(1)
plt.gca().set_aspect('equal', adjustable='box')
for i in range(1, data_nih['x'].shape[1]):
    p = plt.plot(data_nih['x'][:,i], data_nih['y'][:,i], '.', label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['x'][:,i], data_nb['y'][:,i], '*', label='wh', alpha=0.8, c=p[0].get_color())
plt.xlabel('$x$ [au]')
plt.xlabel('$y$ [au]')
plt.legend()
plt.savefig('compare_coord.pdf')
plt.close(1)

plt.figure(2)
for i in range(1, data_nih['ecc'].shape[1]):
    p = plt.plot(data_nih['ecc'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['ecc'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$e$')
plt.legend()
plt.savefig('compare_ecc.pdf')
plt.close(2)

plt.figure(3)
for i in range(1, data_nih['a'].shape[1]):
    p = plt.plot(data_nih['a'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['a'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$a$')
plt.legend()
plt.savefig('compare_a.pdf')
plt.close(3)

plt.figure(4)
for i in range(1, data_nih['inc'].shape[1]):
    p = plt.plot(data_nih['inc'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['inc'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$i$ [rad]')
plt.legend()
plt.savefig('compare_inc.pdf')
plt.close(4)

wh_nih.buf.recorder.stop()
wh_nih.stop()
