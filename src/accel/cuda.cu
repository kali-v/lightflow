#include "cuda_runtime.h"
#include "stdio.h"
#include <iostream>

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

void matmul_cuda(float* host_a, float* host_b, float* res, int ah, int aw, int bw) {
    size_t a_bytes = ah * aw * sizeof(float);
    size_t b_bytes = aw * bw * sizeof(float);
    size_t c_bytes = ah * bw * sizeof(float);

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, a_bytes);
    cudaMalloc(&dev_b, b_bytes);
    cudaMalloc(&dev_c, c_bytes);

    cudaMemcpy(dev_a, host_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, b_bytes, cudaMemcpyHostToDevice);

    int bs = 32;
    dim3 grids(std::ceil(bw / (float)bs), std::ceil(ah / (float)bs));
    dim3 blocks(bs, bs);

    matmul_cuda_kernel<<<grids, blocks>>>(dev_a, dev_b, dev_c, ah, aw, bw);

    cudaMemcpy(res, dev_c, c_bytes, cudaMemcpyDeviceToHost);
}
