#ifndef CPUACC_H
#define CPUACC_H

void matmul_cpu(float* a, float* b, float* c, int ah, int aw, int bw);
Tensor correlate_cpu(Tensor& x, Tensor& filter, DimVec stride, DimVec padding);

#endif
