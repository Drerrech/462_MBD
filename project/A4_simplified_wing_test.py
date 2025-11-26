import A4_simplified_wing

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os


# x_0 = A4_simplified_wing.unscale_x(torch.tensor([31.816013, 11.131291, 0.394867, 2.839057, -0.059896, 4.384855, 0.457936], dtype=torch.float32))


# print(A4_simplified_wing.scale_x(torch.tensor([0.0, 1.0, 0.11, 0.79, 0.69, 0.63, 0.56])))
# print(A4_simplified_wing.simplified_wing_constrained_scaled(torch.tensor([0.0, 1.0, 0.11, 0.79, 0.69, 0.63, 0.56], dtype=torch.float32)))

# mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/simplified_wing/experiments/v2_I_fb_scaled.txt", delta=1e-2, max_f_evals=300, eps_stop=1e-4)



x_points = [
    A4_simplified_wing.scale_x(torch.tensor([31.189580, 11.764323, 0.308703, 2.263694, 1.846517, 3.514550, 0.554865], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([32.588919, 10.557899, 0.378052, 2.486108, 0.065039, 3.468889, 0.552145], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([32.079102, 11.812559, 0.448913, 2.068371, 2.853186, 3.738923, 0.626907], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([33.527317, 10.616684, 0.314521, 2.318346, -0.015038, 3.395902, 0.430008], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([34.985853, 7.797190, 0.319924, 2.449637, 0.797270, 3.472679, 0.306466], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([33.364984, 10.965593, 0.453209, 2.808944, -0.404625, 4.164341, 0.397238], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([31.715910, 9.548441, 0.463347, 2.771223, 2.998771, 4.070483, 0.590728], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([32.474976, 11.752060, 0.445037, 1.331712, 2.904655, 4.325306, 0.462528], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([36.343230, 10.923098, 0.321047, 2.138781, 2.467584, 4.535766, 0.415086], dtype=torch.float32)),
    A4_simplified_wing.scale_x(torch.tensor([39.973077, 11.476375, 0.452780, 2.647991, -0.411234, 4.775192, 0.533311], dtype=torch.float32))
]
# rescale
x_points = [A4_simplified_wing.unscale_x(i) for i in x_points]

from tqdm import tqdm

log_path = "project/logs/simplified_wing/simplified_wing_data_v2/"

for x in tqdm(x_points):
    os.makedirs(log_path+str(x.tolist()), exist_ok=True)
    
    
    print("[I] gradient:")
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_q_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("basic grad finished")
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_q_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("v2 grad finished")

    print("[I -I] gradient:")
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("basic grad finished")
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("v2 grad finished")

    print("[I] \ ei gradient:")
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_basic(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("basic grad finished")
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("v2 grad finished")
    
    print("random grad: ")
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_random_fb_300.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_random_q_300.txt", max_f_evals=300, eps_stop=-1)
    # mbd.mbd_v2(A4_simplified_wing.simplified_wing_constrained_scaled, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("v2 random finished")
