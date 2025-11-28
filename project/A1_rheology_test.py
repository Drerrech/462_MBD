import A1_rheology

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os

x_points = [
    torch.tensor([15.0,          20.0,          10.0]),
    torch.tensor([8.172517606,   5.058263716,   5.444856567]),
    torch.tensor([13.04832254,   15.84400309,   9.950620587]),
    torch.tensor([12.31453665,   13.75028434,   9.557207957]),
    torch.tensor([11.36633281,   12.12935162,   8.906909739]),
    torch.tensor([9.690281657,   6.799833301,   5.904578444]),
    torch.tensor([12.20082785,   12.61627174,   8.890182552]),
    torch.tensor([10, 15, 10], dtype=torch.float32),
    torch.tensor([10, 10, 10], dtype=torch.float32),
    torch.tensor([5, 10, 5], dtype=torch.float32)
]

from tqdm import tqdm

log_path = "project/logs/rheology/rheology_data_v2/"

for x in tqdm(x_points):
    os.makedirs(log_path+str(x.tolist()), exist_ok=True)
    
    
    print("[I] gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_fb_300_plain.txt", max_f_evals=300, eps_stop=-1)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_fb_300_plain.txt", max_f_evals=300, eps_stop=-1)
    print("v2 grad finished")

    print("[I -I] gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_fb_300_plain.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_fb_300_plain.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity)
    print("v2 grad finished")

    print("[I] \ ei gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_fb_300_plain.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_fb_300_plain.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut)
    print("v2 grad finished")

    print("random grad: ")
    mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_random_fb_300_plain.txt", max_f_evals=300, eps_stop=-1)
    mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_random_q_300_plain.txt", max_f_evals=300, eps_stop=-1)
    # mbd.mbd_v2(A1_rheology.rheology_4_sum, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=-1)
    print("v2 random finished")


    print("==sum of models:")
    print("[I] gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, f_post_process=A1_rheology.rheology_post_processing)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, f_post_process=A1_rheology.rheology_post_processing)
    print("v2 grad finished")

    print("[I -I] gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity, f_post_process=A1_rheology.rheology_post_processing)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_double_identity, f_post_process=A1_rheology.rheology_post_processing)
    print("v2 grad finished")

    print("[I] \ ei gradient:")
    mbd.mbd_basic(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut, f_post_process=A1_rheology.rheology_post_processing)
    print("basic grad finished")
    mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_simplex_grad_sum_of_models, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, get_D=models.get_D_identity_random_cut, f_post_process=A1_rheology.rheology_post_processing)
    print("v2 grad finished")
    
    print("random grad: ")
    mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_random_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_random_fb_300_sum.txt", max_f_evals=300, eps_stop=-1, f_post_process=A1_rheology.rheology_post_processing)
    mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_random_q_300_sum.txt", max_f_evals=300, eps_stop=-1, f_post_process=A1_rheology.rheology_post_processing)
    # mbd.mbd_v2(A1_rheology.rheology_4_element_wise, x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=-1, f_post_process=A1_rheology.rheology_post_processing)
    print("v2 random finished")
