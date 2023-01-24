#include <fstream>
#include <iostream>
#include <ostream>

#include "fashion_mnist_dataset.h"
#include <lightflow/config.h>
#include <lightflow/loss.h>
#include <lightflow/nn.h>
#include <lightflow/optimizer.h>
#include <lightflow/tensor.h>

int main(int argc, char** argv) {
    // note that the dataset in ./data is only a snippet not a full dataset
    FashionMnistDataset dataset =
        FashionMnistDataset("data/fashion_mnist_train_vectors.csv", "data/fashion_mnist_train_labels.csv");

    Sequential model =
        Sequential({new Conv2D(1, 24, {3, 3}, {3, 3}, {1, 1}), new LeakyReLU(), new MaxPool2D(2),
                    new Conv2D(24, 32, {2, 2}, {1, 1}, {0, 0}), new LeakyReLU(), new MaxPool2D(2),
                    new Conv2D(32, 48, {2, 2}, {2, 2}, {1, 1}), new LeakyReLU(), new Flatten(), new Linear(192, 10)});

    Adam sgd = Adam(model.parameters());

    Tensor running_loss = Tensor::scalar(0);
    float accuracy = 0;
    int bs = 100;

    for (int i = 0; i < dataset.len(); i++) {
        std::vector<Tensor> data = dataset.get_item(i);
        Tensor input = data[0];
        Tensor label = data[1];

        sgd.zero_grad();

        Tensor pred = model(input);
        Tensor loss = softmax_cross_entropy_loss(&pred, &label);
        loss.backward();

        sgd.step();

        running_loss += loss;
        if (pred.argmax() == label.argmax()) {
            accuracy++;
        }

        if (i % bs == 0 && i > 0) {
            std::cout << i << " train_loss: " << running_loss.data[0] / bs << " train_acc: " << accuracy / bs
                      << std::endl;
            running_loss = Tensor::scalar(0);
            accuracy = 0;
        }
    }
}