
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

void matmul_avx(float* a, float* b, float* c, int ah, int aw, int bw) {
    const int BSB = 64;
    const int BSA = 4;

#pragma omp parallel for
    for (int bb = 0; bb < ah; bb += BSB) {
        float bbm = std::min(bb + BSB, ah);
        for (int ba = 0; ba < aw; ba += BSA) {
            float bam = std::min(ba + BSA, aw);
            for (int i = bb; i < bbm; i++) {
                for (int j = ba; j < bam; j++) {
                    __m256 vec_a = _mm256_set1_ps(a[i * aw + j]);

                    int k;
                    for (k = 0; k <= bw - 8; k += 8) {
                        _mm256_storeu_ps(&c[i * bw + k], _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&b[j * bw + k]),
                                                                         _mm256_loadu_ps(&c[i * bw + k])));
                    }

                    // compute exceding elements
                    for (int q = 0; q + k < bw; q++) {
                        c[i * bw + k + q] += a[i * aw + j] * b[j * bw + k + q];
                    }
                }
            }
        }
    }
}

Tensor correlate_avx(Tensor& x, Tensor& filter, DimVec stride, DimVec padding) {
    int fil_height = filter.dshape[0];
    int fil_width = filter.dshape[1];
    int x_height = x.dshape[0];
    int x_width = x.dshape[1];
    int x_channels = x.shape[1];

    int nrows = floor((x_height - fil_height + 2 * padding[0]) / stride[0] + 1);
    int ncols = floor((x_width - fil_width + 2 * padding[1]) / stride[1] + 1);

    int fil_size = fil_height * fil_width;
    int x_size = x_height * x_width;
    int out_size = nrows * ncols;

    Tensor res = Tensor({x.shape[0], filter.shape[0], nrows, ncols}, 0.0f);

    for (int xn = 0; xn < x.shape[0]; xn++) {
#pragma omp parallel for
        for (int n = 0; n < filter.shape[0]; n++) {
            for (int ch = 0; ch < x_channels; ch++) {
                for (int i = 0; i < nrows; i++) {
                    float* rtmp_data = &res.data.data()[xn * filter.shape[0] * out_size + n * out_size + i * nrows];
                    int ps0 = (i - padding[0]) * stride[0];
                    for (int x_row = std::max(0, ps0); x_row < x_height && x_row < fil_height + ps0; x_row++) {
                        int f_off = n * x_channels * fil_size + ch * fil_size + (x_row - ps0) * fil_height;
                        int x_off = xn * x_channels * x_size + ch * x_size + x_width * x_row;
                        for (int j = 0; j < ncols; j++) {
                            int ps1 = (j - padding[1]) * stride[1];
                            int x_col;
                            __m256 resvec = _mm256_setzero_ps();
                            for (x_col = std::max(0, ps1); x_col < x_width - 8 && x_col - ps1 < fil_width - 8;
                                 x_col += 8) {
                                resvec = _mm256_fmadd_ps(_mm256_loadu_ps(&x.data[x_off + x_col]),
                                                         _mm256_loadu_ps(&filter.data[f_off - ps1 + x_col]), resvec);
                            }
                            for (; x_col < x_width && x_col - ps1 < fil_width; x_col++) {
                                rtmp_data[j] += x.data[x_off + x_col] * filter.data[f_off - ps1 + x_col];
                            }
                            rtmp_data[j] += resvec[0] + resvec[1] + resvec[2] + resvec[3] + resvec[4] + resvec[5] +
                                            resvec[6] + resvec[7];
                        }
                    }
                }
            }
        }
    }

    return res;
}