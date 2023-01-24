#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fashion_mnist_dataset.h"
#include <lightflow/tensor.h>

FashionMnistDataset::FashionMnistDataset(const char* xfile_path, const char* yfile_path, int limit) {
    std::string line;
    std::ifstream xfile(xfile_path);
    std::ifstream yfile(yfile_path);

    while (std::getline(xfile, line) && ((int)this->x.size() < limit || limit == -1)) {
        std::stringstream lss(line);
        std::string item;
        std::vector<float> tensor_data;

        while (std::getline(lss, item, ',')) {
            tensor_data.push_back(std::stof(item) / (float)255);
        }

        this->x.push_back(Tensor({28, 28}, tensor_data));
    }
    xfile.close();

    while (std::getline(yfile, line) && ((int)this->y.size() < limit || limit == -1)) {
        int category = std::stoi(line);
        std::vector<float> label_data;
        for (int i = 0; i < 10; i++) {
            label_data.push_back(category == i);
        }

        y.push_back(Tensor({10}, label_data));
    }

    yfile.close();
}

std::vector<Tensor> FashionMnistDataset::get_item(int index) { return {this->x[index], this->y[index]}; }

int FashionMnistDataset::len() { return this->y.size(); }
