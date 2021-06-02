

import torch
import numpy as np

from nn_models import MLP

class NIH(torch.nn.Module):
    """
    Neural Interacting Hamiltonian (NIH) for the Wisdom-Holman integrator.
    """
    def __init__(self, input_dim, differentiable_model, assume_canonical_coords=True, device='cpu'):
        super(NIH, self).__init__()
        self.device = device
        self.differentiable_model = differentiable_model.to(self.device)
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) 

    def forward(self, x):
        y = self.differentiable_model(x) # call dnn prediction
        assert y.dim() == 2 and y.shape[1] == 1, "Output tensor should have shape [batch_size, 1]"
        return y


    def time_derivative(self, x, t=None, separate_fields=False):

        # outputs an energy-like quantities. 
        # x = x.to(self.device)
        F = self.forward(x) # traditional forward pass

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0] # gradients for conservative field
        conservative_field = dF @ torch.eye(*self.M.shape, device=self.device)

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n, device=self.device)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n, device=self.device) # matrix of ones
            M *= 1 - torch.eye(n, device=self.device) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M


