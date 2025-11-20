import torch
import subprocess
import time

# Step 1: Wake all GPUs (by allocating a trivial tensor on each)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        x = torch.randn(1, device=f"cuda:{i}")
        _ = x * 1
time.sleep(1)

# Step 2: Record power draw for 2 minutes
readings = []
for _ in range(600):  # 120 seconds
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    ).decode("utf-8").strip().split("\n")
    powers = [float(p) for p in output if p]
    readings.append(powers)
    time.sleep(1)

# Step 3: Compute average per GPU
readings_T = list(zip(*readings))  # transpose list of lists
avg_per_gpu = [sum(r) / len(r) for r in readings_T]

# Print results
for i, avg in enumerate(avg_per_gpu):
    print(f"GPU {i}: {avg:.2f} W average idle consumption")
