import torch

def gen_simplex_grad(D, delta_f):
    D_t_pinv = torch.linalg.pinv(D.t())
    return D_t_pinv @ delta_f
