
import torch, argparse
import numpy as np
from config import CONFIG
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from nih import NIH
from data import get_dataset
from utils import custom_loss, get_model_path, get_backbone


def train(config):
    # set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # init model and optimizer
    if config['verbose']:
        print("Training the Neural Interacting Hamiltonian (NIH)...")

    output_dim = 1
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    nn_model = get_backbone(config['backbone']).to(device)
    model = NIH(config['input_dim'], differentiable_model=nn_model, device=device).to(device)
    print(model)
    optim = torch.optim.Adam(model.parameters(), config['learn_rate'], weight_decay=0)

    # arrange data
    data = get_dataset(config['name'], config['data_dir'], verbose=True)
    x = torch.tensor(data['coords'].reshape(data['coords'].shape[0]*data['coords'].shape[1],6), requires_grad=True, dtype=torch.float32, device=device)
    test_x = torch.tensor(data['test_coords'].reshape(data['test_coords'].shape[0]*data['test_coords'].shape[1],6), requires_grad=True, dtype=torch.float32, device=device)
    dxdt = torch.Tensor(data['dcoords'].reshape(data['dcoords'].shape[0]*data['dcoords'].shape[1],6)).cuda()
    test_dxdt = torch.Tensor(data['test_dcoords'].reshape(data['test_dcoords'].shape[0]*data['test_dcoords'].shape[1],6)).cuda()

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(config['total_steps']+1):

        # train step
        ixs = torch.randperm(x.shape[0], device=device)[:config['batch_size']]
        dxdt_hat = model.time_derivative(x[ixs])
        loss = custom_loss(dxdt[ixs], dxdt_hat, x[ixs])
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
        optim.step() ; optim.zero_grad()

        # run test data
        test_ixs = torch.randperm(test_x.shape[0], device=device)[:config['batch_size']]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs])
        test_loss = custom_loss(test_dxdt[test_ixs], test_dxdt_hat, test_x[test_ixs])

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if config['verbose'] and step % config['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
                .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

    return model, stats

if __name__ == "__main__":
    config = CONFIG
    print(config)
    model, stats = train(config)

    # save
    os.makedirs(config['save_dir']) if not os.path.exists(config['save_dir']) else None
    torch.save(model.state_dict(), get_model_path())
    
    for param in model.differentiable_model.parameters():
        print(param.data)
    
