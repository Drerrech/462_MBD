import A2_runge_kutta

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os

# x_0 = torch.tensor([-2, -2], dtype=torch.float32)

# mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/basic_I_fb.txt", max_f_evals=500, eps_stop=-1)
# mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x_0, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/basic_I_q.txt", max_f_evals=500, eps_stop=-1)

# mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/v2_I_fb.txt", delta=1e-2, max_f_evals=500, eps_stop=-1)
# mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x_0, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/v2_I_q.txt", delta=1e-2, max_f_evals=500, eps_stop=-1)


x_points = [
    torch.tensor([-2, -2], dtype=torch.float32),
    torch.tensor([-10, -10], dtype=torch.float32),
    torch.tensor([-5, -20], dtype=torch.float32),
    torch.tensor([-1, -1], dtype=torch.float32),
    torch.tensor([1, 1], dtype=torch.float32),
    torch.tensor([20, 20], dtype=torch.float32),
    torch.tensor([20, 60], dtype=torch.float32),
    torch.tensor([5, 60], dtype=torch.float32),
    torch.tensor([-1, -60], dtype=torch.float32),
    torch.tensor([0.5, 1.5], dtype=torch.float32),
    torch.tensor([-0.5, -1.5], dtype=torch.float32),
    torch.tensor([5, 5], dtype=torch.float32),
    torch.tensor([-5, -5], dtype=torch.float32),
    torch.tensor([0.8, 5], dtype=torch.float32),
    torch.tensor([0.4, 5], dtype=torch.float32),
    torch.tensor([0.9, 7], dtype=torch.float32)
]
x_points = [
    torch.tensor([54, 66, 86, 8, 29, 51, 32, 15], dtype=torch.float32),
    torch.tensor([77.7, 19.1, 44.3, 24.8, 80.6, 22.7, 38.4, 16.], dtype=torch.float32),
    torch.tensor([51.8, 72.6, 68.4, 7.1, 16.2, 39.7, 48.8, 36.6], dtype=torch.float32),
    torch.tensor([62.1, 36.4, 86.8, 2.5, 6.2, 49.5, 41.9, 16.6], dtype=torch.float32),
    torch.tensor([79.6, 78.7, 29.0, 55.2, 19.1, 19.7, 45.6, 6.2], dtype=torch.float32),
    torch.tensor([62.0, 21.3, 96.5, 93.2, 59.9, 23.2, 34.9, 11.5], dtype=torch.float32),
    torch.tensor([56.7, 47.3, 72.9, 5.7, 8.3, 22.5, 56.3, 7.8], dtype=torch.float32),
    torch.tensor([88.1, 73.8, 52.6, 6.6, 6.2, 22.2, 45.3, 33.6], dtype=torch.float32),
    torch.tensor([63.8, 57.2, 59.1, 0.5, 19.5, 55.2, 38.3, 17.1], dtype=torch.float32),
    torch.tensor([66.0, 32.7, 44.6, 48.0, 82.2, 22.3, 57.4, 0.6], dtype=torch.float32)
]

from tqdm import tqdm

log_path = "project/logs/runge_kutta/runge_kutta_data_v2/"

for x in tqdm(x_points):
    os.makedirs(log_path+str(x.tolist()), exist_ok=True)
    
    
    print("[I] gradient:")
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_q_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("basic grad finished")
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_q_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("v2 grad finished")

    print("[I -I] gradient:")
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("basic grad finished")
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("v2 grad finished")

    print("[I] \ ei gradient:")
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("basic grad finished")
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("v2 grad finished")
    
    print("random grad: ")
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_random_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_random_q_300.txt", max_f_evals=300, eps_stop=-1)
    # mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrained, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("v2 random finished")
