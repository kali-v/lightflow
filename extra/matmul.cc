// g++ -Wall -Ofast -funroll-all-loops -mavx -mfma matmul.cc -o matmul

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

const int AH = 2048;
const int AW = 2048;
const int BW = 2048;

void matmul(float* a, float* b, float* c) {
    for (int i = 0; i < AH; i++) {
        for (int j = 0; j < BW; j++) {
            for (int k = 0; k < AW; k++) {
                c[i * BW + j] += a[i * AW + k] * b[k * BW + j];
            }
        }
    }
}

void matmul2(float* a, float* b, float* c) {
    for (int i = 0; i < AH; i++) {
        for (int k = 0; k < AW; k++) {
            float av = a[i * AW + k];
            for (int j = 0; j < BW; j++) {
                c[i * BW + j] += av * b[k * BW + j];
            }
        }
    }
}

void matmul3(float* a, float* b, float* c) {
    const int BSA = 32;
    const int BSB = 512;

    for (int ba = 0; ba < AW; ba += BSA) {
        for (int i = 0; i < AH; i++) {
            for (int bb = 0; bb < BW; bb += BSB) {
                for (int k = ba; k < std::min(ba + BSA, AW); k++) {
                    float av = a[i * AW + k];
                    for (int j = bb; j < std::min(bb + BSB, BW); j++) {
                        c[i * BW + j] += av * b[k * BW + j];
                    }
                }
            }
        }
    }
}

void matmul4(float* a, float* b, float* res) {
    for (int i = 0; i < AH; i++) {
        for (int j = 0; j < AW; j++) {
            __m256 vec_a = _mm256_set1_ps(a[i * AW + j]);
            int k;
            for (k = 0; k < BW - 8; k += 8) {
                _mm256_storeu_ps(&res[i * BW + k], _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&b[j * BW + k]),
                                                                   _mm256_loadu_ps(&res[i * BW + k])));
            }
            // compute exceding elements
            for (int q = 0; q + k < BW; q++) {
                // std::cout << "asd" << std::endl;
                res[i * BW + k + q] += a[i * AW + j] * b[j * BW + k + q];
            }
        }
    }
}

void matmul5(float* a, float* b, float* res) {
    const int BSB = 64;
    const int BSA = 4;

    for (int bb = 0; bb < AH; bb += BSB) {
        float bbm = std::min(bb + BSB, AH);
        for (int ba = 0; ba < AW; ba += BSA) {
            float bam = std::min(ba + BSA, AW);
            for (int i = bb; i < bbm; i++) {
                for (int j = ba; j < bam; j++) {
                    __m256 vec_a = _mm256_set1_ps(a[i * AW + j]);

                    int k;
                    for (k = 0; k <= BW - 8; k += 8) {
                        _mm256_storeu_ps(&res[i * BW + k], _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&b[j * BW + k]),
                                                                           _mm256_loadu_ps(&res[i * BW + k])));
                    }

                    // compute exceding elements
                    for (int q = 0; q + k < BW; q++) {
                        res[i * BW + k + q] += a[i * AW + j] * b[j * BW + k + q];
                    }
                }
            }
        }
    }
}

void fill_matrix(float* mat, int s) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};

    for (int i = 0; i < s; i++) {
        mat[i] = d(gen);
    }
}

int main() {
    float* a = new float[AH * AW];
    float* b = new float[AW * BW];
    float* c = new float[AH * BW];

    fill_matrix(a, AH * AW);
    fill_matrix(b, AW * BW);

    int dur_total = 0;
    int iter = 10;

    for (int i = 0; i < iter; i++) {
        auto st = std::chrono::steady_clock::now();
        matmul5(a, b, c);
        auto et = std::chrono::steady_clock::now();
        int dur = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        dur_total += dur;

        double gflops = (2.0 * AW * AH * BW) * 1e-9 / (dur * 1e-3);

        std::cout << gflops << " GFLOP/s ----  " << dur << " ms" << std::endl;
    }
    std::cout << dur_total << std::endl;

    if (iter != 1)
        exit(0);

    // matmul check
    float* d = new float[AH * BW];
    matmul2(a, b, d);

    for (int i = 0; i < AH * BW; i++) {
        if (std::abs(c[i] - d[i]) > 0.001) {
            std::cout << "assert failed on " << i << std::endl;
            std::cout << c[i] << std::endl;
            std::cout << d[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "assert ok" << std::endl;
}
