#include <cmath>

#include "tensor.h"

#ifndef NN_H
#define NN_H

inline float relu(float x) { return (x > 0) * x; }

inline float leaky_relu(float x, float negative_slope) { return std::max(.0f, x) + negative_slope * std::min(.0f, x); }

inline float sigmoid(float x) {
    float den = 1 + std::pow(std::exp(1.0), -x);
    return 1 / den;
}

class Weight {
  public:
    Tensor* weight_;
    Tensor* t_weight_; // transposed weight

    Weight(Tensor weight);

    void operator=(Tensor& weight);
    Tensor& operator()(bool transpose = true);

    Tensor grad(bool from_transposed = true);
};

class Module {
  public:
    Tensor* operator()(Tensor* x);

    virtual std::vector<Tensor*> parameters();
    virtual Tensor* forward(Tensor* x);
};

class Linear : public Module {
  public:
    int in_features;
    int out_features;

    Tensor* bias;
    Weight* weight;

    Linear(int in_features, int out_features);

    Tensor* forward(Tensor* x);

    std::vector<Tensor*> parameters();
};

class Padding : public Module {
  public:
    DimVec padding;
    float value = 0.0f;

    Padding(DimVec padding, float value = 0.0f);

    Tensor* forward(Tensor* x);
};

class Flatten : public Module {
  public:
    Flatten();

    Tensor* forward(Tensor* x);
};

class MaxPool2D : public Module {
  public:
    int kernel_size;
    MaxPool2D(int kernel_size);

    Tensor* forward(Tensor* x);
};

class LeakyReLU : public Module {
  public:
    float negative_slope;

    LeakyReLU(float negative_slope = 0.01f);

    Tensor* forward(Tensor* x);
};

class ReLU : public Module {
  public:
    ReLU();
    Tensor* forward(Tensor* x);
};

class Sigmoid : public Module {
  public:
    Sigmoid();
    Tensor* forward(Tensor* x);
};

class Conv2D : public Module {
  public:
    int in_channels;
    int out_channels;

    DimVec kernel_size;
    DimVec stride;
    DimVec padding;

    Padding* padding_layer;

    Tensor* bias;
    Weight* weight;

    Conv2D(int in_channels, int out_channels, DimVec kernel_size, DimVec stride = {1}, DimVec padding = {0});

    Tensor* forward(Tensor* x);

    std::vector<Tensor*> parameters();
};

typedef std::vector<Module*> ModuleRef;
class Sequential {
  public:
    std::vector<Tensor*> layers_input;
    ModuleRef layers;

    Sequential(ModuleRef layers);

    Tensor& operator()(Tensor& x);
    std::vector<Tensor*> parameters();
};

#endif
