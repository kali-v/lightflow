#ifndef CPUACC_H
#define CPUACC_H

void check_cpu(const char* fc_name, const Device device);

void matmul_cpu(float* a, float* b, float* c, int ah, int aw, int bw);
void _matmul_deep_cpu(Tensor& a, Tensor& b, float* res, MatmulFunc mm);

Tensor correlate_cpu(Tensor& x, Tensor& filter, DimVec stride, DimVec padding);

#endif
