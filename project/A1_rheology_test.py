import A1_rheology

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial

x_0 = torch.tensor([11.36633281, 12.12935162, 8.906909739], dtype=torch.float32)
x_1 = torch.tensor([8.172517606, 5.058263716, 5.444856567], dtype=torch.float32)
x_2 = torch.tensor([13.04832254, 15.84400309, 9.950620587], dtype=torch.float32)

mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x_0, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path="project/logs/rheology/testing_random_descent/v2_sum_I_fb_375_plain.txt", f_post_process=A1_rheology.rheology_post_processing, max_f_evals=375)
# mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x_0, models.gen_random_grad, line_search.forward_backward_line_search, log_path="project/logs/rheology/testing_random_descent/v2_sum_I_fb_375_random.txt", f_post_process=A1_rheology.rheology_post_processing, max_f_evals=375)
mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x_0, models.gen_simplex_grad_sum_of_models, line_search.quadratic_interpolation_line_search, log_path="project/logs/rheology/testing_random_descent/v2_sum_I_qshift_375_plain.txt", f_post_process=A1_rheology.rheology_post_processing, max_f_evals=375)
# mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x_0, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/rheology/testing_random_descent/v2_sum_I_q_375_random.txt", f_post_process=A1_rheology.rheology_post_processing, max_f_evals=375)
