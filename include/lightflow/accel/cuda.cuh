#ifndef CUDA_CUH
#define CUDA_CUH

void matmul_cuda(float* a, float* other, float* res, int ah, int aw, int bw);
void matmul_deep_cuda(float* a, float* b, float* res, int ah, int aw, int bw, int ch, int bs);

bool compare_arrays_cuda(float* a, float* b, float threshold, int size);

#endif