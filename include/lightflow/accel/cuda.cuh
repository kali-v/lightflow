#ifndef CUDA_CUH
#define CUDA_CUH

void matmul_cuda(float* a, float* other, float* res, int ah, int aw, int bw);
bool compare_arrays_cuda(float* a, float* b, float threshold, int size);

#endif