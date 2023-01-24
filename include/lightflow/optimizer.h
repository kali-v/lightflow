#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

#include "tensor.h"

class Adam {
  public:
    std::vector<Tensor*> parameters;

    float a;
    float b1;
    float b2;

    std::vector<Tensor> m;
    std::vector<Tensor> v;

    int t;
    float eps;

    Adam(std::vector<Tensor*> parameters, float a = 0.001, float b1 = 0.9, float b2 = 0.999);

    void step();
    void zero_grad();
};

class SGD {
  private:
    std::vector<Tensor> diff;

  public:
    std::vector<Tensor*> parameters;
    float momentum;
    float lr;

    SGD(std::vector<Tensor*> parameters, float lr, float momentum);

    void zero_grad();
    void step();
};

#endif
