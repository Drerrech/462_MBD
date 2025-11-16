import torch

gamma = torch.tensor([
0.0137,
0.0274,
0.0434,
0.0866,
0.137,
0.274,
0.434,
0.866,
1.37,
2.74,
4.34,
5.46,
6.88
])

tao = torch.tensor([
3220,
2190,
1640,
1050,
766,
490,
348,
223,
163,
104,
76.7,
68.1,
58.2
])


def eps_sq(x): # x: eta_0, lambda, betea
    eps_no_abs = x[0] * (1 + x[1]**2 * torch.pow(gamma, 2))**((x[2]-1)/2) - tao
    return eps_no_abs**2


def rheology_4_element_wise(x):
    x_c = x.clone()
    
    x_c[0] = 520 * x[0]
    x_c[1] = 14 * x[1]
    x_c[2] = 0.038 * x[2]
    return eps_sq(x_c)


def rheology_post_processing(y):
    return torch.sum(y, dim=0)


def rheology_4_sum(x):
    return rheology_post_processing(rheology_4_element_wise(x)).item()