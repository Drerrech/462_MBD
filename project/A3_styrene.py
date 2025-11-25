import subprocess
import torch
import random


def styrene_unconstrained(x, dir_path="project/problem_executables_M4"): # given that x is a tensor
    # step 1: write the point to a txt file in order to pass it to the executable
    with open(dir_path + "/styrene_point.txt", "w") as f:
        s = " ".join(map(str, x.tolist()))
        f.write(s)
    
    # step 2: call black box
    result = subprocess.run(
        [dir_path + "/styrene", dir_path + "/styrene_point.txt"],
        capture_output=True,
        text=True
    )

    # step 3: convert string output to numerical and tensor form
    # check for errors
    if len(result.stderr) != 0:
        print(f"ERROR: at x={x.tolist()} styrene encountered an error: {result.stderr}")
    
    try:
        values = torch.tensor([float(x) for x in result.stdout.split()], dtype=torch.float32)
    except:
        print(f"ERROR: at x={x.tolist()} std has: {result.stdout}")
        return torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return values


def styrene_constrained(x, log_constraint_violations=True):
    vs = styrene_unconstrained(x)
    v = vs[11].item()

    if log_constraint_violations:
        if torch.any(vs[0:4]): # at least one of the first 4 flags is non-zero -> unrelaxible constraint violation
            print(f"WARNING: at {x.tolist()} one of the first 4 flags were true, output values are meaningless! | f = {v}")
        
        if torch.any(vs[4:11] > 0):
            print(f"warning: at {x.tolist()} got relaxable contraint violation | f = {v}")
            v = min(v, 0)
    
    if v == 1e20: # simulation failed
        v = 0 # as all values are negative, this is suboptimal however as solutions reach -1e7, perhaps rescale output so mbd gradient doesn't explode? TODO
        if log_constraint_violations:
            print(f"WARNING: at {x.tolist()} the simulation failed!")
    
    if v != 1:
        return v
    else:
        return random.random()


def styrene_constrained_scaled_output(x, log_constraint_violations=True):
    v = styrene_constrained(x, log_constraint_violations=log_constraint_violations)
    # given that v is in [~-1e7 0]
    if v <= 0:
        v_scaled = v * 1e-6
    else:
        v_scaled = v
    return v_scaled



def styrene_surrogate_unconstrained(x, dir_path="project/problem_executables_M4"): # given that x is a tensor
    # step 1: write the point to a txt file in order to pass it to the executable
    with open(dir_path + "/styrene_surrogate_point.txt", "w") as f:
        s = " ".join(map(str, x.tolist()))
        f.write(s)
    
    # step 2: call black box
    result = subprocess.run(
        [dir_path + "/styrene_surrogate", dir_path + "/styrene_surrogate_point.txt"],
        capture_output=True,
        text=True
    )

    # step 3: convert string output to numerical and tensor form
    # check for errors
    if len(result.stderr) != 0:
        print(f"ERROR: at x={x.tolist()} styrene_surrogate encountered an error: {result.stderr}")
    
    try:
        values = torch.tensor([float(x) for x in result.stdout.split()], dtype=torch.float32)
    except:
        print(f"ERROR: at x={x.tolist()} std has: {result.stdout}")
        return torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return values


def styrene_surrogate_constrained(x, log_constraint_violations=True):
    vs = styrene_surrogate_unconstrained(x)
    v = vs[11].item()

    if log_constraint_violations:
        if torch.any(vs[0:4]): # at least one of the first 4 flags is non-zero -> unrelaxible constraint violation
            print(f"WARNING: at {x.tolist()} one of the first 4 flags were true, output values are meaningless! | f = {v}")
        
        if torch.any(vs[4:11] > 0):
            print(f"warning: at {x.tolist()} got relaxable contraint violation | f = {v}")
            v = min(v, 0)
    
    if v == 1e20: # simulation failed
        v = 0 # as all values are negative, this is suboptimal however as solutions reach -1e7, perhaps rescale output so mbd gradient doesn't explode? TODO
        if log_constraint_violations:
            print(f"WARNING: at {x.tolist()} the simulation failed!")
    
    if v != 1:
        return v
    else:
        return random.random()


def styrene_surrogate_constrained_scaled_output(x, log_constraint_violations=True):
    v = styrene_surrogate_constrained(x, log_constraint_violations=log_constraint_violations)
    # given that v is in [~-1e7 0]
    if v <= 0:
        v_scaled = v * 1e-6
    else:
        v_scaled = v
    return v_scaled