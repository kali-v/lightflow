#include <chrono>
#include <iostream>

#include <lightflow/tensor.h>

int matmul(DimVec a_shape, DimVec b_shape, int it) {
    Tensor a = Tensor::random(a_shape);
    Tensor b = Tensor::random(b_shape);

    float dur = 0;
    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor c = a.matmul(b);
        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
    }
    return dur;
}

int corr(DimVec a_shape, DimVec b_shape, int it) {
    Tensor a = Tensor::random(a_shape);
    Tensor b = Tensor::random(b_shape);

    float dur = 0;
    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor c = a.correlate(b);
        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
        //std::cout << c.to_string() << std::endl;
    }
    return dur;
}

int main() {
    std::cout << matmul({4096, 4096}, {4096, 4096}, 1) << std::endl;
    std::cout << matmul({2048, 2048}, {2048, 2048}, 5) << std::endl;
    std::cout << matmul({5, 5, 2048, 2048}, {5, 5, 2048, 2048}, 1) << std::endl;
    std::cout << corr({4096, 4096}, {8, 8}, 1) << std::endl;
    std::cout << corr({1024, 1024}, {64, 64}, 1) << std::endl;
}
