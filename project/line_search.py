import torch

def forward_backward_line_search(f, x_k, d_k, g_approx_k, armijo_n, n_max=1e3, f_post_process=lambda y: y):
    # 0) init
    tao = 1
    # n_max initialised in function header
    line_search_success = False

    f_at_x_k = f_post_process(f(x_k))

    # 1) f-b search
    if f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_n * tao * torch.dot(d_k, g_approx_k):
        line_search_success = True
        while tao < 2**n_max and f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_n * tao * torch.dot(d_k, g_approx_k):
            tao *= 2
        tao /= 2
    else:
        line_search_success = False
        while tao >= 2**-n_max and f_post_process(f(x_k + tao * d_k)) >= f_at_x_k + armijo_n * tao * torch.dot(d_k, g_approx_k):
            tao /= 2
        
        if f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_n * tao * torch.dot(d_k, g_approx_k):
            line_search_success = True
    
    # 2) output
    if line_search_success:
        return tao
    else:
        return -1 # fail