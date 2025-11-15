import subprocess
import time

t = time.time_ns()

for i in range(100):
    result = subprocess.run(
        ["./styrene", "points/points_styrene/30_feasible_pts.txt"],     # program + arguments
        capture_output=True,
        text=True
    )

    print("Output:", result.stdout)
    print("Errors:", result.stderr)

print("s / f_c:", (time.time_ns() - t) * 1e-9 / 100)