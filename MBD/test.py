import mbd
import models
import line_search

import torch

def f(x):
    return torch.sum(torch.pow(x, 2), dim=0).item()

def get_D(delta, n_dim, seed=0):
        return delta * torch.eye(n_dim)
        #return delta * torch.cat((torch.eye(n_dim), -torch.eye(n_dim)), dim=1)

torch.set_printoptions(profile="default")
mbd.mbd_basic(f, torch.ones(1), models.gen_simplex_grad, line_search.forward_backward_line_search, get_D)

# p = mbd.point_reuse(f)

# print(p.evaluate(torch.ones(1)))
# print(p.evaluate(torch.ones(1)))