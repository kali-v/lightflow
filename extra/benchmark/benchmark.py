# taskset -c 0 python benchmark.py

import time
import numpy as np
from scipy.signal import correlate


def matmul(a_shape, b_shape, it):
    def _matmul():
        a = np.random.rand(*a_shape)
        b = np.random.rand(*b_shape)
        dur = 0
        for _ in range(it):
            st = time.monotonic()
            a @ b
            dur += time.monotonic() - st
        return int(dur * 1e3)
    name = f"matmul_{str(a_shape)}_{str(b_shape)}"
    return name, _matmul


def corr(a_shape, b_shape, it):
    def _corr():
        a = np.random.rand(*a_shape)
        b = np.random.rand(*b_shape)

        dur = 0
        for _ in range(it):
            st = time.monotonic()
            correlate(a, b, mode="valid")
            dur += time.monotonic() - st
        return int(dur * 1e3)
    name = f"corr_{str(a_shape)}_{str(b_shape)}"
    return name, _corr


with open('lfres') as f:
    lfres = f.read().splitlines()

tasks = [
    matmul((4096, 4096), (4096, 4096), 1),
    matmul((2048, 2048), (2048, 2048), 5),
    matmul((5, 5, 2048, 2048), (5, 5, 2048, 2048), 1),
    corr((4096, 4096), (8, 8), 1),
    corr((1024, 1024), (64, 64), 1)
]

for i, [name, task] in enumerate(tasks):
    py_dur = task()
    lf_dur = int(lfres[i])
    diff = py_dur/lf_dur
    color = '\033[92m' if diff > 0.7 else '\033[91m'
    print(f"---\n{name}:\n{color}{diff:.2f} lf_dur: {lf_dur}ms\tpy_dur: {py_dur}ms\033[0m")

