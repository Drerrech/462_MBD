import models

import torch

class point_reuse:
    def __init__(self, f):
        self.f = f
        self.f_points = {}
        self.points_raw = []
    
    def evaluate(self, x):
        x_hash = x.numpy().tobytes()

        if x_hash in self.f_points.keys(): # already evaluated at this exact point
            return self.f_points[x_hash]
        else: # must evaluate from scratch
            val = self.f(x)
            self.f_points[x_hash] = val
            self.points_raw.append(x)
            return val # 1 stands for 1 evaluation of the function
    
    def get_n_f_evals(self):
        return len(self.f_points)


def mbd_basic(f, x, grad_approx, line_search, log_path, delta=1, min_delta=1e-8, target_acc=1, armijo_eta=0.05, eps_d=0.1, eps_stop=1e-4, max_f_evals=1e16, get_D=models.get_D_identity, f_post_process=lambda y: y):
    open(log_path, "w").close() # clear log file
    with open(log_path, "a") as _f: # add columns
        _f.write("k    | x                                                                                | f(x)                             | delta  | target_acc | ||~g||         | f_evals | success | msg\n")
    def log_progress(msg=""):
        s = f"{k:4} | {str([round(i, 2) for i in x.tolist()]):80} | {f_post_process(f_val_at_x):32.2f} | {delta:6.4f} | {target_acc:10.4f} | {norm_g_approx:14.2f} | {f_evals:7} | {success:7} | {msg}"

        with open(log_path, "a") as _f:
            _f.write(s + "\n")
    
    
    # 0) init
    p_reuse = point_reuse(f)

    N_DIM = x.shape[0]

    k = 0
    max_k = 1e3

    success = False

    norm_g_approx = -1

    f_evals = 0
    f_val_at_x = p_reuse.evaluate(x)
    f_evals = p_reuse.get_n_f_evals()

    log_progress()
    
    while True:
        cur_msg = ""
    # 1) model the gradient at xk
        # build (gen) grad
        f_val_at_x = f(x)
        g_approx = grad_approx(N_DIM, x, p_reuse, delta, f_val_at_x, get_D)
    
    
    # 2) model accuracy checks
        # a)
        norm_g_approx = torch.norm(g_approx)
        if delta < eps_stop and norm_g_approx < eps_stop:
            success = True
            log_progress(msg="declared success in step 2a and terminated")
            return
        # b)
        skip_to_5 = False
        if delta > target_acc * norm_g_approx:
            # insufficient accuracy
            delta = max(0.5 * delta, min_delta) # NOTE: up to modification
            skip_to_5 = True
            cur_msg = "2b triggered"
    

        if not skip_to_5:
    # 3) line-search
            # select descent direction
            d = -g_approx # NOTE: up to modification
            
            # perform line search
            t = line_search(p_reuse.evaluate, x, d, g_approx, armijo_eta, f_post_process=f_post_process, delta=delta)
            
    # 4) update
            if t != -1: # line search success
                x = x + t*d # NOTE: up to modification
                cur_msg += "line search success"
            else: # line search failure
                target_acc = 0.5 * target_acc # NOTE: up to modification
                cur_msg += "line search failure"
            
            # update f_evals
            f_evals = p_reuse.get_n_f_evals()


    # 5) termination test
        if k == max_k or f_evals >= max_f_evals: # TODO: add a non-dogshit termination test
            log_progress(msg="termination test triggered, algorithm terminated")
            return
        else:
            k += 1
       
    # log progress (no termination needed it should be handled in step 2a or 5)
        log_progress(msg=cur_msg)


# improved: delta update, descent direction, post-line serach check (optional)
def mbd_v2(f, x, grad_approx, line_search, log_path, delta=1, min_delta=1e-8, target_acc=1, armijo_eta=0.05, eps_d=0.1, eps_stop=1e-4, max_f_evals=1e16, f_post_process=lambda y: y, get_D=models.get_D_identity, check_d_post_line_search=False):
    open(log_path, "w").close() # clear log file
    with open(log_path, "a") as _f: # add columns
        _f.write("k    | x                                                                                | f(x)                             | delta  | target_acc | ||~g||         | f_evals | success | msg\n")
    def log_progress(msg=""):
        s = f"{k:4} | {str([round(i, 2) for i in x.tolist()]):80} | {f_post_process(f_val_at_x):32.2f} | {delta:6.4f} | {target_acc:10.4f} | {norm_g_approx:14.2f} | {f_evals:7} | {success:7} | {msg}"

        with open(log_path, "a") as _f:
            _f.write(s + "\n")
    
    
    # 0) init
    p_reuse = point_reuse(f)

    N_DIM = x.shape[0]

    k = 0
    max_k = 1e3

    success = False

    norm_g_approx = -1

    f_evals = 0
    f_val_at_x = p_reuse.evaluate(x)
    f_evals = p_reuse.get_n_f_evals()

    log_progress()
    
    while True:
        cur_msg = ""
    # 1) model the gradient at xk
        # build (gen) grad
        f_val_at_x = f(x)
        g_approx = grad_approx(N_DIM, x, p_reuse, delta, f_val_at_x, get_D)
    
    
    # 2) model accuracy checks
        # a)
        norm_g_approx = torch.norm(g_approx)
        if delta < eps_stop and norm_g_approx < eps_stop:
            success = True
            log_progress(msg="declared success in step 2a and terminated")
            return
        # b)
        skip_to_5 = False
        if delta > target_acc * norm_g_approx:
            # insufficient accuracy
            delta = max(min(0.5 * delta, 0.5 * target_acc * norm_g_approx), min_delta) # NOTE: up to modification
            skip_to_5 = True
            cur_msg = "2b triggered"
    

        if not skip_to_5:
    # 3) line-search
            # select descent direction
            d = -g_approx / norm_g_approx # NOTE: up to modification
            
            # perform line search
            t = line_search(p_reuse.evaluate, x, d, g_approx, armijo_eta, f_post_process=f_post_process, delta=delta)
            
    # 4) update
            if t != -1: # line search success
                if check_d_post_line_search:
                    # check if line search is actually better than f-evals used for the gradient
                    D = get_D(delta, N_DIM, f=f, x_k=x)
                    p = D.shape[1]
                    
                    min_f_val_grad = f_post_process(p_reuse.evaluate(x + D[:, 0]))
                    min_f_val_grad_idx = 0
                    for i in range(1, p): # for generalised simplex grad
                        if min_f_val_grad > f_post_process(p_reuse.evaluate(x + D[:, i])):
                            min_f_val_grad = f_post_process(p_reuse.evaluate(x + D[:, i]))
                            min_f_val_grad_idx = i
                    
                    new_f = f_post_process(p_reuse.evaluate(x + t*d))
                    if new_f < min_f_val_grad:
                        x = x + t*d # NOTE: up to modification
                    else:
                        x = x + D[:, min_f_val_grad_idx]
                else:
                    x = x + t*d # NOTE: up to modification
                
                cur_msg += "line search success"
            else: # line search failure
                target_acc = 0.5 * target_acc # NOTE: up to modification
                cur_msg += "line search failure"
            
            # update f_evals
            f_evals = p_reuse.get_n_f_evals()


    # 5) termination test
        if k == max_k or f_evals >= max_f_evals: # TODO: add a non-dogshit termination test
            log_progress(msg="termination test triggered, algorithm terminated")
            return
        else:
            k += 1
       
    # log progress (no termination needed it should be handled in step 2a or 5)
        log_progress(msg=cur_msg)
