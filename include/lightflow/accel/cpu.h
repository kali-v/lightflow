#ifndef CPUACC_H
#define CPUACC_H

#include "tensor.h"

inline float relu_cpu(float x) { return (x > 0) * x; }
inline float sigmoid_cpu(float x) { return 1 / (1 + std::exp(-x)); }
inline float leaky_relu_cpu(float x, float negative_slope) { return std::max(.0f, x) + negative_slope * std::min(.0f, x); }


void check_cpu(const char* fc_name, const Device device);

void matmul_cpu(float* a, float* b, float* c, int ah, int aw, int bw);
void _matmul_deep_cpu(Tensor& a, Tensor& b, float* res, MatmulFunc mm);

Tensor correlate_cpu(Tensor& x, Tensor& filter, DimVec stride, DimVec padding);

Tensor transpose_cpu(Tensor& a);

#endif
