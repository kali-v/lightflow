#include <cmath>

#include "tensor.h"

#ifndef NN_H
#define NN_H

class Weight {
  public:
    Tensor* weight_;
    Tensor* t_weight_; // transposed weight

    Weight(Tensor weight);
    ~Weight();

    void operator=(Tensor& weight);
    Tensor& operator()(bool transpose = true);

    Tensor grad(bool from_transposed = true);
};

class Module {
  public:
    Tensor* operator()(Tensor* x);

    virtual ~Module();

    virtual std::vector<Tensor*> parameters();
    virtual Tensor* forward(Tensor* x);
};

class Linear : public Module {
  public:
    int in_features_;
    int out_features_;

    Tensor* bias_;
    Weight* weight_;

    Linear(int in_features, int out_features);
    ~Linear();

    Tensor* forward(Tensor* x);

    std::vector<Tensor*> parameters();
};

class Padding : public Module {
  public:
    DimVec padding_;
    float value_ = 0.0f;

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
    int kernel_size_;
    MaxPool2D(int kernel_size);

    Tensor* forward(Tensor* x);
};

class LeakyReLU : public Module {
  public:
    float negative_slope_;

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
    int in_channels_;
    int out_channels_;

    DimVec kernel_size_;
    DimVec stride_;
    DimVec padding_;

    Padding* padding_layer_;

    Tensor* bias_;
    Weight* weight_;

    Conv2D(int in_channels, int out_channels, DimVec kernel_size, DimVec stride = {1}, DimVec padding = {0});
    ~Conv2D();

    Tensor* forward(Tensor* x);

    std::vector<Tensor*> parameters();
};

typedef std::vector<Module*> ModuleRef;
class Sequential {
  public:
    std::vector<Tensor*> layers_input_;
    ModuleRef layers_;

    Sequential(ModuleRef layers);
    ~Sequential();

    Tensor& operator()(Tensor& x);
    std::vector<Tensor*> parameters();
};

#endif
