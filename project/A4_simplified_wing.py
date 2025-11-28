import subprocess
import torch
import random


def simplified_wing_unconstrained(x, dir_path="project/problem_executables_M4"): # given that x is a tensor
    # step 1: write the point to a txt file in order to pass it to the executable
    with open(dir_path + "/simplified_wing_point.txt", "w") as f:
        s = " ".join(map(str, x.tolist()))
        f.write(s)
    
    # step 2: call black box
    result = subprocess.run(
        [dir_path + "/simplified_wing", dir_path + "/simplified_wing_point.txt"],
        capture_output=True,
        text=True
    )

    # step 3: convert string output to numerical and tensor form
    # check for errors
    if len(result.stderr) != 0:
        print(f"ERROR: at x={x.tolist()} simplified wing encountered an error: {result.stderr}")
        return torch.tensor([0, 0, 0, random.random()], dtype=torch.float32)
    
    try:
        values = torch.tensor([float(x) for x in result.stdout.split()], dtype=torch.float32)
    except:
        print(f"ERROR: at x={x.tolist()} std has: {result.stdout}")
        return torch.tensor([0, 0, 0, random.random()], dtype=torch.float32)
    return values


def scale_x(x):
    x_scaled = x.clone()
    x_scaled[0] = 30 + x[0]*(45-30)
    x_scaled[1] = 6 + x[1]*(12-6)
    x_scaled[2] = 0.28 + x[2]*(0.50-0.28)
    x_scaled[3] = -1 + x[3]*(3+1)
    x_scaled[4] = -1 + x[4]*(3+1)
    x_scaled[5] = 1.6 + x[5]*(5.0-1.6)
    x_scaled[6] = 0.30 + x[6]*(0.79-0.30)
    return x_scaled

def unscale_x(x): # brimmiest naming award
    x_scaled = x.clone()
    x_scaled[0] = (x[0]-30) / 15
    x_scaled[1] = (x[1]-6) / 6
    x_scaled[2] = (x[2]-0.28) / 0.22
    x_scaled[3] = (x[3]+1) / 4
    x_scaled[4] = (x[4]+1) / 4
    x_scaled[5] = (x[5]-1.6) / 3.4
    x_scaled[6] = (x[6]-0.3) / 0.49
    return x_scaled

def simplified_wing_unconstrained_scaled(x):
    # 45 12 0.50 3 3 5.0 0.79
    # 30 6 0.28 −1 −1 1.6 0.30
    # scale each from [0, 1] to the proper bounds

    return simplified_wing_unconstrained(scale_x(x))

def simplified_wing_constrained_scaled(x, log_constraint_violations=True):
    vs = simplified_wing_unconstrained_scaled(x)
    v = vs[3].item()

    if torch.any(vs[0:3] > 0): # at least one of the first 4 flags is non-zero -> unrelaxible constraint violation
        if log_constraint_violations:
            print(f"warning: at {x.tolist()} constrain violation | f = {v}")
    
    if v == 1e20: # simulation failed
        v = 1 # as all values are negative, this is suboptimal however as solutions reach -1e7, perhaps rescale output so mbd gradient doesn't explode? TODO
        if log_constraint_violations:
            print(f"WARNING: at {x.tolist()} the simulation failed!")

    # check if boundaries are violated
    if torch.any(x < 0.001) or torch.any(x > 0.999): # violaltion!
        v = 0
        for xi in x:
            if xi < 0.001 or xi > 0.999:
                v += (xi-0.5)**2
        return v
    
    # check if constraints are violated
    if torch.any(vs[0:3] > 0): # violaltion!
        v = 0
        for vi in vs[0:3]:
            v += vi**2
        return v
    
    
    if v != 1:
        return v
    else:
        return random.random()
