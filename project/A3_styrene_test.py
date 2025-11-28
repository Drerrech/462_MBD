import A3_styrene

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial
import os

#A3_styrene.parse_log_to_true("project/logs/styrene/experiments_surrogate/v2_I_fb_scaled.txt", "project/logs/styrene/experiments_surrogate/v2_I_fb_scaled_sur_to_true.txt")

# best: 99.999953599999997778, 94.029649999999946886, 95.15957000000000221, 0.04209999999999999853, 0, 50.142644800000006455, 32.61161999999998784, 51.59949660000002325
#90.5, 90, 90, 10, 10, 50, 32, 51
# x_0 = torch.tensor([50.0, 66.0, 86.0, 8.0, 29.0, 51.0, 32.0, 15.0], dtype=torch.float32)

# print(A3_styrene.styrene_constrained_scaled_output(torch.tensor([55.17, 66.03, 86.11, 8.02, 27.58, 51.38, 32.0, 14.78], dtype=torch.float32)))

# best: -11338700.00
# mbd.mbd_v2(A3_styrene.styrene_surrogate_constrained, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/styrene/experiments_surrogate/v2_I_fb.txt", delta=1, max_f_evals=300, eps_stop=1e-4)

# best: -11.55e6 (better) on some x: [54.0, 66.0, 86.0, 8.0, 29.0, 51.0, 32.0, 15.0] -> [55.03, 65.35, 86.0, 8.0, 28.34, 51.1, 32.0, 15.54] 
# surrogate: -10.97e6 same x ->
#mbd.mbd_v2(A3_styrene.styrene_surrogate_constrained_scaled_output, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/styrene/experiments_surrogate/v2_I_fb_scaled.txt", delta=1e1, max_f_evals=300, eps_stop=1e-4, log_decimal_pts=8)

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
    torch.tensor([66.0, 32.7, 44.6, 48.0, 82.2, 22.3, 57.4, 0.6], dtype=torch.float32),
    torch.tensor([49.0, 55.8, 80.2, 64.3, 23.9, 44.3, 58.6, 33.6] , dtype=torch.float32),
    torch.tensor([65.5, 63.3, 59.2, 19.1, 1.1 , 18.8, 48.7, 0.1 ], dtype=torch.float32),
    torch.tensor([82.2, 11.4, 61.4, 57.7, 68.2, 28.5, 51.5, 23.0] , dtype=torch.float32),
    torch.tensor([58.3, 73.9, 21.6, 97.9, 38.8, 24.6, 44.5, 19.3], dtype=torch.float32),
    torch.tensor([71.5, 42.2, 79.8, 28.8, 41.1, 51.3, 43.9, 6.8 ], dtype=torch.float32),
    torch.tensor([49.0, 75.5, 57.5, 17.9, 17.7, 58.5, 35.0, 0.5 ], dtype=torch.float32),
    torch.tensor([51.4, 71.1, 71.4, 93.6, 4.5 , 33.6, 48.6, 40.1] , dtype=torch.float32),
    torch.tensor([90.4, 5.3,  66.6, 14.3, 71.1, 39.87, 40.6, 21.0] , dtype=torch.float32),
    torch.tensor([92.1, 20.8, 81.5, 86.7, 21.1, 51.0, 40.0, 13.2] , dtype=torch.float32),
    torch.tensor([72.8, 51.0, 96.6, 35.7, 41.2, 57.7, 57.2, 6.6 ], dtype=torch.float32)
]

from tqdm import tqdm

log_path = "project/logs/styrene/styrene_surrogate_data_v2/"

# for x in tqdm(x_points):
#     os.makedirs(log_path+str(x.tolist()), exist_ok=True)
    
    
#     print("[I] gradient:")
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_fb_300.txt", max_f_evals=300, eps_stop=1e-7)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]_q_300.txt", max_f_evals=300, eps_stop=1e-7)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]_qv_300.txt", max_f_evals=300, eps_stop=1e-7)
#     print("basic grad finished")
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_fb_300.txt", max_f_evals=300, eps_stop=1e-7)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]_q_300.txt", max_f_evals=300, eps_stop=1e-7)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]_qv_300.txt", max_f_evals=300, eps_stop=1e-7)
#     print("v2 grad finished")

#     print("[I -I] gradient:")
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I-I]_q_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     print("basic grad finished")
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_fb_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I-I]_q_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I-I]_qv_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_double_identity)
#     print("v2 grad finished")

#     print("[I] \ ei gradient:")
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     mbd.mbd_basic(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/basic_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     print("basic grad finished")
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_fb_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_q_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_[I]-ei_qv_300.txt", max_f_evals=300, eps_stop=1e-7, get_D=models.get_D_identity_random_cut)
#     print("v2 grad finished")
    
#     print("random grad: ")
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_random_grad, line_search.forward_backward_line_search, log_path=log_path+str(x.tolist())+"/v2_random_fb_300.txt", max_f_evals=300, eps_stop=1e-7)
#     mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_random_grad, line_search.quadratic_interpolation_line_search, log_path=log_path+str(x.tolist())+"/v2_random_q_300.txt", max_f_evals=300, eps_stop=1e-7)
#     # mbd.mbd_v2(partial(A3_styrene.styrene_surrogate_constrained_scaled_output, log_constraint_violations=False), x, models.gen_random_grad, line_search.quadratic_interpolation_line_search_voodo, log_path=log_path+str(x.tolist())+"/v2_random_qv_300.txt", max_f_evals=300, eps_stop=1e-7)
#     print("v2 random finished")


for entry in tqdm(os.listdir(log_path)):
        subdir = os.path.join(log_path, entry)
        
        if os.path.isdir(subdir):
            for file in os.listdir(subdir):
                if file.endswith(".txt"):
                    full_path = os.path.join(subdir, file)
                    
                    dest_path = "project/logs/styrene/styrene_truth_data_v2/" + os.path.join(subdir, file)
                    os.makedirs("project/logs/styrene/styrene_truth_data_v2/" + subdir, exist_ok=True)
                    A3_styrene.parse_log_to_true(full_path, dest_path)
