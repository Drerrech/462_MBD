import subprocess
import torch


def runge_kutta_unconstrained(x, dir_path="project/problem_executables_M4"): # given that x is a tensor
    # step 1: write the point to a txt file in order to pass it to the executable
    with open(dir_path + "/runge_kutta_point.txt", "w") as f:
        s = " ".join(map(str, x.tolist()))
        f.write(s)
    
    # step 2: call black box
    result = subprocess.run(
        [dir_path + "/rungekutta", dir_path + "/runge_kutta_point.txt"],
        capture_output=True,
        text=True
    )

    # step 3: convert string output to numerical and tensor form
    # check for errors
    if len(result.stderr) != 0:
        print(f"ERROR: at x={x.tolist()} runge kutta encountered an error: {result.stderr}")
    
    value = float(result.stdout)
    return value


def runge_kutta_constrianed(x):
    v = runge_kutta_unconstrained(x)

    if v == 1e20: # constraing
        v = 1e2
    
    return v
