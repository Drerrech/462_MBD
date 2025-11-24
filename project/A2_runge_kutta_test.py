import A2_runge_kutta

import mbd as mbd
import models as models
import line_search as line_search

import torch
from functools import partial

x_0 = torch.tensor([-2, -2], dtype=torch.float32)

mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/basic_I_fb.txt", max_f_evals=500, eps_stop=-1)
mbd.mbd_basic(A2_runge_kutta.runge_kutta_constrianed, x_0, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/basic_I_q.txt", max_f_evals=500, eps_stop=-1)

mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x_0, models.gen_simplex_grad, line_search.forward_backward_line_search, log_path="project/logs/runge_kutta/v2_I_fb.txt", delta=1e-2, max_f_evals=500, eps_stop=-1)
mbd.mbd_v2(A2_runge_kutta.runge_kutta_constrianed, x_0, models.gen_simplex_grad, line_search.quadratic_interpolation_line_search, log_path="project/logs/runge_kutta/v2_I_q.txt", delta=1e-2, max_f_evals=500, eps_stop=-1)
