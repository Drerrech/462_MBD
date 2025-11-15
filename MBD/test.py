import mbd
import models
import line_search

import torch

def f(x):
    return torch.sum(torch.pow(x, 2), dim=0).item()

# torch.set_printoptions(profile="default")
# mbd.mbd_basic(f, torch.ones(1), models.gen_simplex_grad, line_search.forward_backward_line_search)

p = mbd.point_reuse(f)

print(p.evaluate(torch.ones(1)))
print(p.evaluate(torch.ones(1)))