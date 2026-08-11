import numpy as np

d = np.load("outputs/benchmark_twophase/twophase_fields.npz")
print("keys:", d.files)
for k in d.files:
    a = d[k]
    print(f"{k:<32} shape={a.shape}")
