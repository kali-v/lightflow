#ifndef CUDA_CUH
#define CUDA_CUH

void move_data_to_cuda(const float* host_ptr, const int host_size, float** dev_ptr);
void move_data_to_host(float* host_ptr, const int host_size, const float* dev_ptr);

bool compare_arrays_cuda(const float* a, const float* b, const float threshold, const int size);

void matmul_cuda(const float* a, const float* other, float* res, const int ah, const int aw, const int bw);
void matmul_deep_cuda(const float* a, const float* b, float* res, const int ah, const int aw, const int bw,
                      const int ch, const int bs);

#endif