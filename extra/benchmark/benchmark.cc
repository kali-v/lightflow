#include <chrono>
#include <iostream>

#include <lightflow/nn.h>
#include <lightflow/tensor.h>
#include <lightflow/loss.h>

int matmul(DimVec a_shape, DimVec b_shape, int it) {
    float dur = 0;
    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor a = Tensor::random(a_shape);
        Tensor b = Tensor::random(b_shape);
        Tensor c = a.matmul(b);
        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
    }
    return dur;
}

int conv2d_load(DimVec a_shape, int it) {
    float dur = 0;
    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor a = Tensor::random(a_shape);
        Sequential model = Sequential({new Conv2D(1, 32, {3, 3}, {3, 3}, {1, 1}), new LeakyReLU(), new MaxPool2D(2),
                                       new Conv2D(32, 128, {2, 2}, {1, 1}, {0, 0}), new LeakyReLU(), new MaxPool2D(2),
                                       new Conv2D(128, 64, {2, 2}, {2, 2}, {1, 1}), new LeakyReLU(), new Flatten(),
                                       new Linear(7744, 512), new LeakyReLU(), new Linear(512, 10)});

        Tensor c = model(a);
        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
    }

    return dur;
}

int conv2d(DimVec a_shape, int it) {
    float dur = 0;
    Tensor a = Tensor::random(a_shape);
    Sequential model = Sequential({new Conv2D(1, 32, {3, 3}, {3, 3}, {1, 1}), new LeakyReLU(), new MaxPool2D(2),
                                   new Conv2D(32, 128, {2, 2}, {1, 1}, {0, 0}), new LeakyReLU(), new MaxPool2D(2),
                                   new Conv2D(128, 64, {2, 2}, {2, 2}, {1, 1}), new LeakyReLU(), new Flatten(),
                                   new Linear(7744, 512), new LeakyReLU(), new Linear(512, 10)});

    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor c = model(a);
        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
    }

    return dur;
}


int linear(DimVec a_shape, int it) {
    float dur = 0;
    Tensor a = Tensor::random(a_shape);
    Sequential model = Sequential({
        new Linear(4096, 512), new LeakyReLU(),
        new Linear(512, 256), new ReLU(),
        new Linear(256, 50), new Sigmoid()
    });

    for (int i = 0; i < it; i++) {
        auto st = std::chrono::steady_clock::now();
        Tensor c = model(a);
        Tensor l = Tensor::random({50});
        Tensor loss = softmax_cross_entropy_loss(&c, &l);
        loss.backward();

        auto et = std::chrono::steady_clock::now();
        dur += std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count();
    }

    return dur;
}

int main() {
    int mul = atoi(getenv("LF_DEFDEV")) == 0 ? 1 : 2;

    std::cout << matmul({4096, 4096}, {4096, 4096}, 1 * mul) << std::endl;
    std::cout << matmul({2048, 2048}, {2048, 2048}, 5 * mul) << std::endl;
    std::cout << matmul({2, 2, 2048, 2048}, {2, 2, 2048, 2048}, 1 * mul) << std::endl;
    std::cout << linear({4096}, 100 * mul) << std::endl;
    std::cout << conv2d_load({1, 1, 256, 256}, 100 * mul) << std::endl;
    std::cout << conv2d({1, 1, 256, 256}, 100 * mul) << std::endl;
}
