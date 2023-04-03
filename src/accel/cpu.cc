#include <algorithm>

void matmul_cpu(float* a, float* b, float* c, int ah, int aw, int bw) {
    const int BSA = 32;
    const int BSB = 512;

    //#pragma omp parallel for
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
