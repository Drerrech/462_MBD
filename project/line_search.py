import torch

def forward_backward_line_search(f, x_k, d_k, g_approx_k, armijo_eta, n_max=1e2, f_post_process=lambda y: y, delta=1):
    # 0) init
    tao = 1
    # n_max initialised in function header
    line_search_success = False

    f_at_x_k = f_post_process(f(x_k))

    # 1) f-b search
    if f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_eta * tao * torch.dot(d_k, g_approx_k):
        line_search_success = True
        while tao < 2**n_max and f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_eta * tao * torch.dot(d_k, g_approx_k):
            tao *= 2
        tao /= 2
    else: 
        while tao >= 2**-n_max and f_post_process(f(x_k + tao * d_k)) >= f_at_x_k + armijo_eta * tao * torch.dot(d_k, g_approx_k):
            tao /= 2
        
        if f_post_process(f(x_k + tao * d_k)) < f_at_x_k + armijo_eta * tao * torch.dot(d_k, g_approx_k):
            line_search_success = True
    
    # 2) output
    if line_search_success:
        return tao
    else:
        return -1 # fail


def quadratic_interpolation_line_search(f, x_k, d_k, g_approx_k, armijo_eta, max_search_it=8, max_tune_it=1, f_post_process=lambda y: y, delta=1, l3_start=1):
    f_at_x_k = f_post_process(f(x_k))

    l1 = 0
    l2 = 0.5*l3_start # TODO: reuse points in some way instead of this garbage o algo
    l3 = l3_start # TODO: improve? TODO: add proper bound

    # step 1: finding appropriate points
    for i in range(max_search_it):
        f1 = f_post_process(f(x_k + l1*d_k))
        f2 = f_post_process(f(x_k + l2*d_k))
        f3 = f_post_process(f(x_k + l3*d_k))

        if f1 <= f2: # shift l2 closer to f1
            l2 *= 0.5
            continue

        # f1 > f2
        if f2 >= f3: #shift l3 further
            l3 *= 2
            continue
        
        break

    f1 = f_post_process(f(x_k + l1*d_k))
    f2 = f_post_process(f(x_k + l2*d_k))
    f3 = f_post_process(f(x_k + l3*d_k))
    
    if (f1 <= f2) or (f2 >= f3):
        return -1
    

    # step 2: optimising further, helping pass armijo
    for i in range(max_tune_it):
        denum = (2*((l2 - l3)*f1 + (l3 - l1)*f2 + (l1 - l2)*f3))
        try:
            l_star = ((l2**2 - l3**2)*f1 + (l3**2 - l1**2)*f2 + (l1**2 - l2**2)*f3) / denum
            if l_star != l_star or (l_star < l1 or l3 < l_star): # nan or outside of bracket
                return -1
        except Exception as e:
            return -1
        
        new_x = x_k + l_star * d_k
        f_at_new_x = f_post_process(f(new_x))

        if l_star < l2:
            if f_at_new_x > f2: # replace l1 with l_star
                l1 = l_star
                f1 = f_post_process(f(x_k + l1*d_k))
            else: # replace l2 with l_star
                l2 = l_star
                f2 = f_post_process(f(x_k + l2*d_k))
        else: # l_star > l2
            if f_at_new_x > f2: # replace l3 with l_star
                l3 = l_star
                f3 = f_post_process(f(x_k + l3*d_k))
            else: # replace l2 with l_star
                l2 = l_star
                f2 = f_post_process(f(x_k + l2*d_k))
    
    # armijo
    if f_at_new_x < f_at_x_k + armijo_eta * l_star * torch.dot(d_k, g_approx_k):
        return l_star
    return -1


def quadratic_interpolation_line_search_voodo(f, x_k, d_k, g_approx_k, armijo_eta, max_search_it=8, max_tune_it=1, f_post_process=lambda y: y, delta=1):
    f_at_x_k = f_post_process(f(x_k))

    l1 = 0
    l2 = 0.5 # TODO: reuse points in some way instead of this garbage o algo
    l3 = 1 # TODO: improve? TODO: add proper bound

    # step 1: finding appropriate points
    for i in range(max_search_it):
        f1 = f_post_process(f(x_k + l1*d_k))
        f2 = f_post_process(f(x_k + l2*d_k))
        if f1 < f2:
            l2 = l1*0.8 + l2*0.2 # TODO: too much voodoo!!! magic numbers!!!
            continue

        f3 = f_post_process(f(x_k + l3*d_k))
        
        if f2 > f3:
            l3 = l3/0.8
            l1 = l1*0.2 + l3*0.8
            l2 = l2*0.2 + l3*0.8
            continue
        
        break
    
    if (f1 < f2) or (f2 > f3):
        return -1
    

    # step 2: optimising further, helping pass armijo
    for i in range(max_tune_it):
        denum = (2*((l2 - l3)*f1 + (l3 - l1)*f2 + (l1 - l2)*f3))
        try:
            l_star = ((l2**2 - l3**2)*f1 + (l3**2 - l1**2)*f2 + (l1**2 - l2**2)*f3) / denum
            if l_star != l_star or (l_star < l1 or l3 < l_star): # nan or outside of bracket
                return -1
        except Exception as e:
            return -1
        
        new_x = x_k + l_star * d_k
        f_at_new_x = f_post_process(f(new_x))

        if l_star < l2:
            if f_at_new_x > f2: # replace l1 with l_star
                l1 = l_star
                f1 = f_post_process(f(x_k + l1*d_k))
            else: # replace l2 with l_star
                l2 = l_star
                f2 = f_post_process(f(x_k + l2*d_k))
        else: # l_star > l2
            if f_at_new_x > f2: # replace l3 with l_star
                l3 = l_star
                f3 = f_post_process(f(x_k + l3*d_k))
            else: # replace l2 with l_star
                l2 = l_star
                f2 = f_post_process(f(x_k + l2*d_k))
    
    # armijo
    if f_at_new_x < f_at_x_k + armijo_eta * l_star * torch.dot(d_k, g_approx_k):
        return l_star
    return -1