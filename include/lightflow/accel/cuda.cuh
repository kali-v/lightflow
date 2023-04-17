#ifndef CUDA_CUH
#define CUDA_CUH

void move_data_to_cuda(const float* host_ptr, const int host_size, float** dev_ptr);
void move_data_to_host(float* host_ptr, const float* dev_ptr, const int size);

bool compare_arrays_cuda(const float* a, const float* b, const float threshold, const int size);

void matmul_cuda(const float* a, const float* other, float* res, const int ah, const int aw, const int bw);
void matmul_deep_cuda(const float* a, const float* b, float* res, const int ah, const int aw, const int bw,
                      const int ch, const int bs);

void transpose_cuda(const float* a, float* res, const int bs, const int ch, const int w, const int h);

void add_cuda(const float* a, const float* b, float* c, int asize, int bsize);
void sub_cuda(const float* a, const float* b, float* c, int asize, int bsize);
void mul_cuda(const float* a, const float* b, float* c, int asize, int bsize);
void div_cuda(const float* a, const float* b, float* c, int asize, int bsize);
void pow_cuda(const float* a, const float* exp, float* c, int size);
void pow_const_cuda(const float* a, const float exp, float* c, int size);
void log_cuda(const float* a, float* b, int size);
void sqrt_cuda(const float* a, float* b, int size);
void exp_cuda(const float* a, float* b, int size);

void relu_cuda(const float* a, float* b, int size);

#endif