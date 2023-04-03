
#include <algorithm>
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
