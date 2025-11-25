import A4_simplified_wing

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os


x_0 = A4_simplified_wing.unscale_x(torch.tensor([31.816013, 11.131291, 0.394867, 2.839057, -0.059896, 4.384855, 0.457936], dtype=torch.float32))

# print(A4_simplified_wing.simplified_wing_constrained_scaled(A4_simplified_wing.unscale_x(torch.tensor([44.19, 6.75, 0.28, 3.0, 0.72, 4.03, 0.3], dtype=torch.float32))))

mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/simplified_wing/experiments/v2_I_fb_scaled.txt", delta=1e-2, max_f_evals=300, eps_stop=1e-4)



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
