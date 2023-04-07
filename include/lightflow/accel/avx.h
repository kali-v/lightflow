#ifndef AVXACC_H
#define AVXACC_H

void matmul_avx(float* a, float* b, float* c, int ah, int aw, int bw);
Tensor correlate_avx(Tensor& x, Tensor& filter, DimVec stride, DimVec padding);

#endif
