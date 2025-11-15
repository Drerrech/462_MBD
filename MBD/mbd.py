import torch

class point_reuse:
    def __init__(self, f):
        self.f = f
        self.f_points = {}
    
    def evaluate(self, x):
        x_hash = x.numpy().tobytes()

        if x_hash in self.f_points.keys(): # already evaluated at this exact point
            return (self.f_points[x_hash], 0)
        else: # must evaluate from scratch
            val = self.f(x)
            self.f_points[x_hash] = val
            return (val, 1) # 1 stands for 1 evaluation of the function


def mbd_basic(f, x, grad_approx, line_search):
    open("results_basic.txt", "w").close() # clear log file
    def log_progress(msg=""):
        s = f"k: {k:4} | x: {str(x.tolist()):32} | f(x): {p_reuse.evaluate(x)[0]:6.2f} | delta: {delta:6.4f} | f_evals: {f_evals:6} | success: {success} | msg: {msg}"

        with open("results_basic.txt", "a") as f:
            f.write(s + "\n")
    
    def get_D():
        return delta * torch.eye(N_DIM)
    
    
    # 0) init
    delta = 1
    target_acc = 1
    armijo_n = 0.05
    eps_d = 0.1 # guessing from this point
    eps_stop = 1e-4

    f_evals = 0
    max_f_evals = 100

    k = 0
    max_k = 100

    p_reuse = point_reuse(f)

    N_DIM = x.shape[0]

    success = False

    
    while True:
    # 0.5 🤤) log progress (no termination needed it should be handled in step 2a or 5)
        log_progress()

    # 1) model the gradient at xk
        # select D
        D = get_D()
        
        # build delta_f
        f_val, _ev = p_reuse.evaluate(x)
        f_evals += _ev
        delta_f = -f_val * torch.ones(N_DIM)
        for i in range(N_DIM):
            f_val, _ev = p_reuse.evaluate(x + D[:, i])
            f_evals += _ev
            delta_f[i] += f_val
        
        # build (gen) grad
        g_approx = grad_approx(D, delta_f)
    
    
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
            delta = 0.5 * delta # NOTE: up to modification
            skip_to_5 = True
    

        if not skip_to_5:
    # 3) line-search
            # select descent direction
            d = -g_approx # NOTE: up to modification
            
            # perform line search
            t = line_search(f, x, d, g_approx, armijo_n)
            
    # 4) update
            if t != -1: # line search success
                x = x + t*d # NOTE: up to modification
            else: # line search failure
                target_acc = 0.5 * target_acc # NOTE: up to modification


        else:
    # 5) termination test
            if k == max_k: # TODO: add a non-dogshit termination test
                log_progress(msg="termination test triggered, algorithm terminatioed")
                return
            else:
                k += 1