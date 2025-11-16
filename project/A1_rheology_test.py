import A1_rheology

import mbd as mbd
import models as models
import line_search as line_search

import torch

x_0 = torch.tensor([9.690281657, 6.799833301, 5.904578444], dtype=torch.float32)

mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x_0, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path="project/A1_results_sum_basic.txt", f_post_process=A1_rheology.rheology_post_processing, max_f_evals=375)
#mbd.mbd_basic(A1_rheology.rheology_4_sum, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/A1_results_basic.txt")


# a = torch.tensor([9.47924518585205, 8.363113403320312, 8.71458911895752])
# print(A1_rheology.rheology_4_sum(a))