import torch

def get_D_identity(delta, n_dim, p_reuse=None, x_k=None):
        return delta * torch.eye(n_dim)

def get_D_generalised_function_reuse(delta, n_dim, p_reuse=None, x_k=None): # f is assumed to be the p_reuse.evaluate
    if p_reuse is None or x_k is None:
        return get_D_identity(delta, n_dim)
    else:
        x_sorted = p_reuse.points_raw.copy()
        x_sorted = sorted(x_sorted, key=lambda xi : torch.norm(xi - x_k))
        directions = []
        for x in x_sorted:
            if torch.norm(x - x_k) <= delta and not torch.equal(x, x_k): # if x is withing the trust region (defined by delta) TODO: up to modification, then add it's direction
                directions.append((x-x_k).unsqueeze(1))
                if len(directions) == n_dim:
                    break
        
        if len(directions) > 0:
            D = torch.cat(directions, dim=1) # add the direction from x_k to x to D
        else:
            D = delta * torch.eye(n_dim)
        # and hope that we have enough lol
        return D


def gen_simplex_grad(N_DIM, x, p_reuse, delta, f_val_at_x, get_D):
    # select D
    D = get_D(delta, N_DIM, p_reuse=p_reuse, x_k=x)
    p = D.shape[1]
    
    # build delta_f
    delta_f = -f_val_at_x * torch.ones(p)
    for i in range(p): # for generalised simplex grad
        f_val = p_reuse.evaluate(x + D[:, i])
        delta_f[i] += f_val
    
    D_t_pinv = torch.linalg.pinv(D.t())
    return D_t_pinv @ delta_f


def gen_simplex_grad_sum_of_models(N_DIM, x, p_reuse, delta, f_val_at_x, get_D): # assuming x is of dim n, f returnx x values/tensors each for a sub-function
    n_models = f_val_at_x.shape[0]
    
    # select D
    D = get_D(delta, N_DIM, p_reuse=p_reuse, x_k=x)
    p = D.shape[1]
    
    
    # build delta_f_matrix: [df1 df2 ...] (df for every sub-function)
    delta_f = -f_val_at_x * torch.ones(p, n_models)
    for i in range(p): # for generalised simplex grad
        f_val = p_reuse.evaluate(x + D[:, i]) # 1-dim n_models size tensor
        delta_f[i, :] += f_val
    
    D_t_pinv = torch.linalg.pinv(D.t())
    
    grad_matrix = D_t_pinv @ delta_f

    sum_of_grads = torch.sum(grad_matrix, dim=1) # TODO: why the fuck is it dim=1
    return sum_of_grads
