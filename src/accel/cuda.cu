#include "stdio.h"
#include <iostream>

#ifdef LF_CUDA_AVAIL
#include "cuda_runtime.h"
#endif

void move_data_to_cuda(const float* host_ptr, const int size, float** dev_ptr) {
    cudaMalloc(dev_ptr, size * sizeof(float));
    cudaMemcpy(*dev_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
}

void move_data_to_host(float* host_ptr, const float* dev_ptr, const int size) {
    cudaMemcpy(host_ptr, dev_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] - b[i];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] * b[i];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = a[i] / b[i];
    }
}

__global__ void pow_kernel(const float* a, const float* exp, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = powf(a[i], exp[0]);
    }
}

__global__ void pow_const_kernel(const float* a, const float exp, float* c, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        c[i] = powf(a[i], exp);
    }
}

__global__ void compare_arrays_kernel(const float* a, const float* b, float* res, const float threshold,
                                      const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (fabs(a[i] - b[i]) > threshold) atomicExch(res, 1.0f);
    }
}

__global__ void matmul_cuda_kernel(const float* a, const float* b, float* c, const int ah, const int aw, const int bw) {
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

__global__ void matmul_deep_cuda_kernel(const float* a, const float* b, float* c, const int ah, const int aw,
                                        const int bw, const int ch, const int bs, const int a_size, const int b_size,
                                        const int c_size) {
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

void add_cuda(const float* a, const float* b, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_kernel<<<num_blocks, block_size>>>(a, b, c, size);
}

void sub_cuda(const float* a, const float* b, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sub_kernel<<<num_blocks, block_size>>>(a, b, c, size);
}

void mul_cuda(const float* a, const float* b, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mul_kernel<<<num_blocks, block_size>>>(a, b, c, size);
}

void div_cuda(const float* a, const float* b, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    div_kernel<<<num_blocks, block_size>>>(a, b, c, size);
}

void pow_const_cuda(const float* a, const float exp, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    pow_const_kernel<<<num_blocks, block_size>>>(a, exp, c, size);
}

void pow_cuda(const float* a, const float* exp, float* c, int size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    pow_kernel<<<num_blocks, block_size>>>(a, exp, c, size);
}

bool compare_arrays_cuda(const float* a, const float* b, const float threshold, const int size) {
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

void matmul_cuda(const float* a, const float* b, float* res, const int ah, const int aw, const int bw) {
    int bs = 32;
    dim3 grids(std::ceil(bw / (float)bs), std::ceil(ah / (float)bs));
    dim3 blocks(bs, bs);

    matmul_cuda_kernel<<<grids, blocks>>>(a, b, res, ah, aw, bw);
}

void matmul_deep_cuda(const float* a, const float* b, float* res, const int ah, const int aw, const int bw,
                      const int ch, const int bs) {
    int block_size = 16;
    dim3 grids(std::ceil(bw / (float)block_size), std::ceil(ah / (float)block_size));
    dim3 blocks(block_size, block_size);

    int a_size = ah * aw;
    int b_size = aw * bw;
    int c_size = ah * bw;

    matmul_deep_cuda_kernel<<<grids, blocks>>>(a, b, res, ah, aw, bw, ch, bs, a_size, b_size, c_size);
}
