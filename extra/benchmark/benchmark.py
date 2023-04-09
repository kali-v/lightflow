import time
import os
import torch
from torch import nn


def matmul(a_shape, b_shape, it):
    def _matmul():
        dur = 0
        for _ in range(it):
            st = time.monotonic()
            a = torch.rand(*a_shape).to(device)
            b = torch.rand(*b_shape).to(device)
            c = a @ b
            dur += time.monotonic() - st
        return int(dur * 1e3)
    shp = (str(a_shape)+str(b_shape)).replace(" ", "")
    return f"mm_{shp}", _matmul


def conv_load(a_shape, it, task_id):
    def _conv():
        dur = 0
        for _ in range(it):
            st = time.monotonic()
            a = torch.rand(*a_shape).to(device)
            model = nn.Sequential(
                nn.Conv2d(1, 32, 3, 3, 1), nn.LeakyReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 128, 2, 1, 0), nn.LeakyReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 64, 2, 2, 1), nn.LeakyReLU(), nn.Flatten(),
                nn.Linear(7744, 512), nn.LeakyReLU(), nn.Linear(512, 10)
            ).to(device)
            model(a)
            dur += time.monotonic() - st
        return int(dur * 1e3)
    return f"conv_load_{task_id}", _conv


def conv(a_shape, it, task_id):
    def _conv():
        dur = 0
        a = torch.rand(*a_shape).to(device)
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 3, 1), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 2, 1, 0), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 2, 2, 1), nn.LeakyReLU(), nn.Flatten(),
            nn.Linear(7744, 512), nn.LeakyReLU(), nn.Linear(512, 10)
        ).to(device)
        for _ in range(it):
            st = time.monotonic()
            model(a)
            dur += time.monotonic() - st
        return int(dur * 1e3)
    return f"conv_{task_id}", _conv


def linear(a_shape, it, task_id):
    def _conv():
        dur = 0
        a = torch.rand(a_shape).to(device)
        model = nn.Sequential(
            nn.Linear(4096, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 50), nn.Sigmoid()
        ).to(device)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(it):
            st = time.monotonic()
            y = model(a)
            l = torch.rand(50).to(device)
            z = loss_fn(y, l)
            z.backward()
            dur += time.monotonic() - st
        return int(dur * 1e3)
    return f"linear_{task_id}", _conv


device = "cpu" if int(os.getenv("LF_DEFDEV")) == 0 else "cuda"
mul = 1 if device == "cpu" else 2

lfres = []
lfpath = "lfres"
if os.path.isfile(lfpath):
    with open(lfpath, "r") as f:
        lfres = f.read().splitlines()

tasks = [
    matmul((4096, 4096), (4096, 4096), 1 * mul),
    matmul((2048, 2048), (2048, 2048), 5 * mul),
    matmul((2, 2, 2048, 2048), (2, 2, 2048, 2048), 1 * mul),
    linear((4096), 100 * mul, 1),
    conv_load((1, 1, 256, 256), 100 * mul, 1),
    conv((1, 1, 256, 256), 100 * mul, 1),
]
py_durs = []
lf_durs = []
for i, [name, task] in enumerate(tasks):
    py_durs.append(task())
    lf_durs.append(int(lfres[i] if i < len(lfres) else 1))
    diff = py_durs[-1]/lf_durs[-1]
    color = '\033[92m' if diff > 0.70 else '\033[91m'
    print(f"{name}:\t{color}{diff:.2f} lf_time: {lf_durs[-1]}ms\tpy_time: {py_durs[-1]}ms\033[0m")

diff = sum(py_durs)/sum(lf_durs)
color = '\033[92m' if diff > 0.70 else '\033[91m'
print(f"overall:\t{color}{diff:.2f} lf_time: {sum(lf_durs)}ms\tpy_time: {sum(py_durs)}ms\033[0m")
