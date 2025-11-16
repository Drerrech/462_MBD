import mbd as mbd
import models as models
import line_search as line_search

import torch

def f(x):
    return torch.sum(torch.pow(x, 2), dim=0).item()

x_0 = torch.tensor([10], dtype=torch.float32)

#mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x_0, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path="project/A1_results_basic.txt", f_post_process=A1_rheology.rheology_post_processing)
mbd.mbd_basic(f, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/test_results_basic.txt")