
#include "tensor.h"
#include <algorithm>
#include <cmath>

void matmul_cpu(float* a, float* b, float* c, int ah, int aw, int bw) {
    const int BSA = 32;
    const int BSB = 512;

#pragma omp parallel for
    for (int ba = 0; ba < aw; ba += BSA) {
        for (int i = 0; i < ah; i++) {
            for (int bb = 0; bb < bw; bb += BSB) {
                for (int k = ba; k < std::min(ba + BSA, aw); k++) {
                    float av = a[i * aw + k];
                    for (int j = bb; j < std::min(bb + BSB, bw); j++) {
                        c[i * bw + j] += av * b[k * bw + j];
                    }
                }
            }
        }
    }
}

Tensor correlate_cpu(Tensor& x, Tensor& filter, DimVec stride, DimVec padding) {
    int fil_height = filter.dshape_[0];
    int fil_width = filter.dshape_[1];
    int x_height = x.dshape_[0];
    int x_width = x.dshape_[1];
    int x_channels = x.shape_[1];

    int nrows = floor((x_height - fil_height + 2 * padding[0]) / stride[0] + 1);
    int ncols = floor((x_width - fil_width + 2 * padding[1]) / stride[1] + 1);

    int fil_size = fil_height * fil_width;
    int x_size = x_height * x_width;
    int out_size = nrows * ncols;

    Tensor res = Tensor({x.shape_[0], filter.shape_[0], nrows, ncols}, 0.0f);

    for (int xn = 0; xn < x.shape_[0]; xn++) {
#pragma omp parallel for
        for (int n = 0; n < filter.shape_[0]; n++) {
            float* rtmp_data = &res.data_.data()[xn * filter.shape_[0] * out_size + n * out_size];
            for (int ch = 0; ch < x_channels; ch++) {
                for (int i = 0; i < nrows; i++) {
                    int ps0 = (i - padding[0]) * stride[0];
                    for (int x_row = std::max(0, ps0); x_row < x_height && x_row < fil_height + ps0; x_row++) {
                        int f_off = n * x_channels * fil_size + ch * fil_size + (x_row - ps0) * fil_height;
                        int x_off = xn * x_channels * x_size + ch * x_size + x_width * x_row;
                        for (int j = 0; j < ncols; j++) {
                            int ps1 = (j - padding[1]) * stride[1];
                            for (int x_col = std::max(0, ps1); x_col < x_width && x_col - ps1 < fil_width; x_col++) {
                                rtmp_data[i * nrows + j] += x.data_[x_off + x_col] * filter.data_[f_off - ps1 + x_col];
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}

void _matmul_deep_cpu(Tensor& a, Tensor& b, float* res, MatmulFunc mm) {
    int a_size = a.dshape_[0] * a.dshape_[1];
    int b_size = b.dshape_[0] * b.dshape_[1];
    int c_size = a.dshape_[0] * b.dshape_[1];

    for (int b0 = 0; b0 < a.shape_[0]; b0++) {
        for (int b1 = 0; b1 < a.shape_[1]; b1++) {
            int tof = b0 * a.shape_[1] * a_size + b1 * a_size;
            int oof = b0 * b.shape_[1] * b_size + b1 * b_size;
            int rof = b0 * b.shape_[1] * c_size + b1 * c_size;

            mm(&a.data_[tof], &b.data_[oof], &res[rof], a.dshape_[0], a.dshape_[1], b.dshape_[1]);
        }
    }
}
