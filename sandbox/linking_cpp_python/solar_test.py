import subprocess
import time

t = time.time_ns()

for i in range(1):
    result = subprocess.run(
        ["./solar", "1", "points/tests_solar/1_MAXNRG_H1/x0.txt"],     # program + arguments
        capture_output=True,
        text=True
    )

    print("Output:", result.stdout)
    print("Errors:", result.stderr)

print("s / f_c:", (time.time_ns() - t) * 1e-9 / 1)