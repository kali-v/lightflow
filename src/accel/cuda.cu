#include "stdio.h"
#include <iostream>

#include "cuda_runtime.h"

__global__ void compare_arrays_kernel(float* a, float* b, float* res, float threshold, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (fabs(a[i] - b[i]) > threshold)
            atomicExch(res, 1.0f);
    }
}

__global__ void matmul_cuda_kernel(float* a, float* b, float* c, int ah, int aw, int bw) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < bw && i < ah) {
        float tmp = 0;
        for (int k = 0; k < aw; k++) {
            tmp += a[i * aw + k] * b[k * bw + j];
        }
        c[i * bw + j] = tmp;
    }
}

void matmul_cuda(float* a, float* b, float* res, int ah, int aw, int bw) {
    size_t c_bytes = ah * bw * sizeof(float);

    float* dev_c;
    cudaMalloc(&dev_c, c_bytes);

    int bs = 32;
    dim3 grids(std::ceil(bw / (float)bs), std::ceil(ah / (float)bs));
    dim3 blocks(bs, bs);

    matmul_cuda_kernel<<<grids, blocks>>>(a, b, dev_c, ah, aw, bw);

    cudaMemcpy(res, dev_c, c_bytes, cudaMemcpyDeviceToHost);
}

bool compare_arrays_cuda(float* a, float* b, float threshold, int size) {
    float* res;
    cudaMallocManaged(&res, sizeof(float));
    res[0] = 0.0;

    const int blocks = 256;
    const int grids = (size + blocks - 1) / blocks;
    compare_arrays_kernel<<<grids, blocks>>>(a, b, res, threshold, size);
    cudaDeviceSynchronize();

    bool result = (*res == 0.0);
    cudaFree(res);

    return result;
}
