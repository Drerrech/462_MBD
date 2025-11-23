import torch

def get_D_identity(delta, n_dim, f=None):
        return delta * torch.eye(n_dim)

def get_D_function_reuse(delta, n_dim, f=None):
    # if f is None:
    #     return get_D_identity(delta, n_dim)
    # else:
    #        for 
    pass

def gen_simplex_grad(N_DIM, x, f, delta, f_val_at_x, get_D=get_D_identity):
    # select D
    D = get_D(delta, N_DIM)
    p = D.shape[1]
    
    # build delta_f
    delta_f = -f_val_at_x * torch.ones(p)
    for i in range(p): # for generalised simplex grad
        f_val = f(x + D[:, i])
        delta_f[i] += f_val
    
    D_t_pinv = torch.linalg.pinv(D.t())
    return D_t_pinv @ delta_f


def gen_simplex_grad_sum_of_models(N_DIM, x, f, delta, f_val_at_x, get_D=get_D_identity): # assuming x is of dim n, f returnx x values/tensors each for a sub-function
    n_models = f_val_at_x.shape[0]
    
    # select D
    D = get_D(delta, N_DIM)
    p = D.shape[1]
    
    
    # build delta_f_matrix: [df1 df2 ...] (df for every sub-function)
    delta_f = -f_val_at_x * torch.ones(p, n_models)
    for i in range(p): # for generalised simplex grad
        f_val = f(x + D[:, i]) # 1-dim n_models size tensor
        delta_f[i, :] += f_val
    
    D_t_pinv = torch.linalg.pinv(D.t())
    
    grad_matrix = D_t_pinv @ delta_f

    sum_of_grads = torch.sum(grad_matrix, dim=1) # TODO: why the fuck is it dim=1
    return sum_of_grads
