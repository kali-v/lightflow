#include "optimizer.h"

#include <iostream>
#include <thread>

#include "loss.h"
#include "nn.h"
#include "tensor.h"

Adam::Adam(std::vector<Tensor*> parameters, float a, float b1, float b2) {
    this->parameters = parameters;

    this->a = a;
    this->b1 = b1;
    this->b2 = b2;

    this->m.reserve(parameters.size());
    this->v.reserve(parameters.size());

    for (Tensor* param : parameters) {
        this->m.push_back(Tensor(param->shape_, 0.0f));
        this->v.push_back(Tensor(param->shape_, 0.0f));
    }

    this->t = 1;
    this->eps = 1e-08;
}

void Adam::step() {
    for (uint i = 0; i < this->parameters.size(); i++) {
        Tensor grad = *this->parameters[i]->grad_;
        Tensor gradp = grad.pow(2.0);
        Tensor mi = this->m[i] * this->b1;
        Tensor mgi = grad * (1 - this->b1);
        this->m[i] = mi + mgi;

        Tensor vi = this->v[i] * this->b2;
        Tensor vgi = gradp * (1 - this->b2);
        this->v[i] = vi + vgi;

        Tensor mht = this->m[i] / (float)(1 - std::pow(this->b1, t));
        Tensor vht = this->v[i] / (float)(1 - std::pow(this->b2, t));
        Tensor vhts = vht.sqrt() + this->eps;

        Tensor diff = mht * this->a / vhts;

        *this->parameters[i] -= diff;
    }
    this->t++;
}

void Adam::zero_grad() {
    for (Tensor* param : this->parameters) {
        param->grad_ = new Tensor(param->shape_, 0, {}, false);
    }
}

SGD::SGD(std::vector<Tensor*> parameters, float lr, float momentum) {
    this->parameters = parameters;
    for (uint i = 0; i < parameters.size(); i++) {
        this->diff.push_back(Tensor::scalar(0));
    }

    this->lr = lr;
    this->momentum = momentum;
}

void SGD::zero_grad() {
    for (Tensor* param : this->parameters) {
        param->grad_ = new Tensor(param->shape_, 0, {}, false);
    }
}

void SGD::step() {
    for (uint i = 0; i < this->parameters.size(); i++) {
        Tensor grad_asc = *this->parameters[i]->grad_ * this->lr;
        diff[i] = diff[i] * this->momentum + grad_asc;

        *this->parameters[i] -= diff[i];
    }
}