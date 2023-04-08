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

__global__ void matmul_deep_cuda_kernel(float* a, float* b, float* c, int ah, int aw, int bw, int ch, int bs,
                                        int a_size, int b_size, int c_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b0 = 0; b0 < bs; b0++) {
        for (int b1 = 0; b1 < ch; b1++) {
            int tof = b0 * ch * a_size + b1 * a_size;
            int oof = b0 * ch * b_size + b1 * b_size;
            int rof = b0 * ch * c_size + b1 * c_size;

            if (j < bw && i < ah) {
                float tmp = 0;
                for (int k = 0; k < aw; k++) {
                    tmp += a[tof + i * aw + k] * b[oof + k * bw + j];
                }
                c[rof + i * bw + j] = tmp;
            }
        }
    }
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

void matmul_cuda(float* a, float* b, float* res, int ah, int aw, int bw) {
    int bs = 32;
    dim3 grids(std::ceil(bw / (float)bs), std::ceil(ah / (float)bs));
    dim3 blocks(bs, bs);

    matmul_cuda_kernel<<<grids, blocks>>>(a, b, res, ah, aw, bw);
}

void matmul_deep_cuda(float* a, float* b, float* res, int ah, int aw, int bw, int ch, int bs) {
    int block_size = 16;
    dim3 grids(std::ceil(bw / (float)block_size), std::ceil(ah / (float)block_size));
    dim3 blocks(block_size, block_size);

    int a_size = ah * aw;
    int b_size = aw * bw;
    int c_size = ah * bw;

    matmul_deep_cuda_kernel<<<grids, blocks>>>(a, b, res, ah, aw, bw, ch, bs, a_size, b_size, c_size);
}
