#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void matmul_cuda(float* a, float* other, float* res, int ah, int aw, int bw);
//void _matmul(float* a, float* other, float* res, int tof, int oof);
