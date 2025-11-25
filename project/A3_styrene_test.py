import A3_styrene

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os

# best: 99.999953599999997778, 94.029649999999946886, 95.15957000000000221, 0.04209999999999999853, 0, 50.142644800000006455, 32.61161999999998784, 51.59949660000002325
#90.5, 90, 90, 10, 10, 50, 32, 51
x_0 = torch.tensor([50.0, 66.0, 86.0, 8.0, 29.0, 51.0, 32.0, 15.0], dtype=torch.float32)

print(A3_styrene.styrene_constrained_scaled_output(torch.tensor([55.17, 66.03, 86.11, 8.02, 27.58, 51.38, 32.0, 14.78], dtype=torch.float32)))

# best: -11338700.00
# mbd.mbd_v2(A3_styrene.styrene_surrogate_constrained, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/styrene/experiments_surrogate/v2_I_fb.txt", delta=1, max_f_evals=300, eps_stop=1e-4)

# best: -11.55e6 (better) on some x: [54.0, 66.0, 86.0, 8.0, 29.0, 51.0, 32.0, 15.0] -> [55.03, 65.35, 86.0, 8.0, 28.34, 51.1, 32.0, 15.54] 
# surrogate: -10.97e6 (shit) samx x ->
# mbd.mbd_v2(A3_styrene.styrene_surrogate_constrained_scaled_output, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/styrene/experiments_surrogate/v2_I_fb_scaled.txt", delta=1e1, max_f_evals=300, eps_stop=1e-4)

# x_points = [
#     torch.tensor([-2, -2], dtype=torch.float32),
#     torch.tensor([-10, -10], dtype=torch.float32),
#     torch.tensor([-5, -20], dtype=torch.float32),
#     torch.tensor([-1, -1], dtype=torch.float32),
#     torch.tensor([1, 1], dtype=torch.float32),
#     torch.tensor([20, 20], dtype=torch.float32),
#     torch.tensor([20, 60], dtype=torch.float32),
#     torch.tensor([5, 60], dtype=torch.float32),
#     torch.tensor([-1, -60], dtype=torch.float32),
#     torch.tensor([0.5, 1.5], dtype=torch.float32),
#     torch.tensor([-0.5, -1.5], dtype=torch.float32),
#     torch.tensor([5, 5], dtype=torch.float32),
#     torch.tensor([-5, -5], dtype=torch.float32),
#     torch.tensor([0.8, 5], dtype=torch.float32),
#     torch.tensor([0.4, 5], dtype=torch.float32),
#     torch.tensor([0.9, 7], dtype=torch.float32)
# ]

# from tqdm import tqdm
# for x in tqdm(x_points):
#     os.makedirs("project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist()), exist_ok=True)
    
#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_I_fb_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_I_q_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_I_qv_300.txt", max_f_evals=300, eps_stop=-1)
#     print("basic grad finished")

#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_random_fb_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_random_q_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/basic_random_qv_300.txt", max_f_evals=300, eps_stop=-1)
#     print("basic random finished")
    
#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_I_fb_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_I_q_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_I_qv_300.txt", max_f_evals=300, eps_stop=-1)
#     print("v2 grad finished")

#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_random_fb_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_random_q_300.txt", max_f_evals=300, eps_stop=-1)
#     mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path="project/logs/runge_kutta/runge_kutta_data_v1/"+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=-1)
#     print("v2 random finished")
