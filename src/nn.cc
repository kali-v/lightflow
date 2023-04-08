#include "nn.h"

#include <cmath>
#include <iostream>
#include <random>

#include "config.h"
#include "diff.h"
#include "loss.h"

float uniform_random() { return (float)rand() / RAND_MAX; }

void xavier_normal_init(Tensor* weights) {
    float fan_in = (float)weights->dshape[0];
    float fan_out = (float)weights->dshape[1];
    float stdev = std::sqrt(2 / (fan_in + fan_out));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd(0, stdev);

    for (Vec1D::iterator it = weights->data.begin(), end = weights->data.end(); it != end; it++) {
        *it = nd(gen);
    }
}

Weight::Weight(Tensor weight) {
    this->t_weight_ = new Tensor(weight.transpose());
    this->t_weight_->requires_grad = true;

    this->weight_ = new Tensor(weight);
    this->weight_->requires_grad = true;
}

Weight::~Weight() {
    delete this->weight_;
    delete this->t_weight_;
}

void Weight::operator=(Tensor& weight) {
    delete this->t_weight_;
    this->t_weight_ = new Tensor(weight.transpose());
    this->t_weight_->requires_grad = true;

    delete this->weight_;
    this->weight_ = new Tensor(weight);
    this->weight_->requires_grad = true;
}

Tensor Weight::grad(bool from_transposed) {
    return from_transposed ? this->t_weight_->grad->transpose() : *this->weight_->grad;
}

Tensor& Weight::operator()(bool transposed) { return transposed ? *this->t_weight_ : *this->weight_; }

Module::~Module() {
    for (Tensor* param : this->parameters()) {
        free(param);
    }
}

Tensor* Module::operator()(Tensor* x) { return forward(x); }

Tensor* Module::forward(Tensor* x) { return x; }

std::vector<Tensor*> Module::parameters() { return {}; }

Linear::Linear(int in_features, int out_features) {
    this->in_features = in_features;
    this->out_features = out_features;

    Tensor weight_tensor = Tensor({out_features, in_features}, 0.0f, {}, true);
    xavier_normal_init(&weight_tensor);

    this->bias = new Tensor({1, out_features}, .1f, {}, true);
    this->weight = new Weight(weight_tensor);
}

Linear::~Linear() {
    delete this->bias;
    delete this->weight;
}

Tensor* Linear::forward(Tensor* x) {
    Tensor* mul_tensor = new Tensor(x->matmul((*this->weight)()));
    Tensor* out_tensor = new Tensor(*mul_tensor + *this->bias);
    return out_tensor;
}

std::vector<Tensor*> Linear::parameters() { return {this->weight->t_weight_, this->bias}; }

Conv2D::Conv2D(int in_channels, int out_channels, DimVec kernel_size, DimVec stride, DimVec padding) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->stride = stride;
    this->padding = padding;

    this->padding_layer = new Padding(padding);

    this->bias = new Tensor({1, out_channels, 1, 1}, 0.1f, {}, true);

    std::vector<int> weight_shape = {out_channels, in_channels, kernel_size[0], kernel_size[1]};
    Tensor weight_tensor = Tensor(weight_shape, 0.0f, {}, true);
    xavier_normal_init(&weight_tensor);

    this->weight = new Weight(weight_tensor);
}

Conv2D::~Conv2D() {
    delete this->padding_layer;
    delete this->bias;
    delete this->weight;
}

Tensor* Conv2D::forward(Tensor* x) {
    Tensor filter = (*this->weight)(false);

    Tensor* padded_x = x;
    if (this->padding[0] > 0 || this->padding[1] > 0) {
        padded_x = this->padding_layer->forward(x);
    }

    Tensor* cor_ten = new Tensor(padded_x->correlate(filter, this->stride));
    cor_ten->children = {padded_x, this->weight->weight_};
    cor_ten->requires_grad = true;
    cor_ten->backward_fn = correlate_backward(padded_x, this->weight->weight_, cor_ten, this->stride);

    return new Tensor(cor_ten->channelwise_sum(*this->bias));
}

std::vector<Tensor*> Conv2D::parameters() { return {this->weight->weight_, this->bias}; }

Padding::Padding(DimVec padding, float value) {
    this->padding = padding;
    this->value = value;
}

Tensor* Padding::forward(Tensor* x) {
    Tensor padded_x = x->pad(this->padding, this->value);

    Tensor* res_tensor = new Tensor(padded_x.shape, padded_x.data, {x}, true);
    res_tensor->backward_fn = pad_backward(x, this->padding, res_tensor);

    return res_tensor;
}

MaxPool2D::MaxPool2D(int kernel_size) { this->kernel_size = kernel_size; }

Tensor* MaxPool2D::forward(Tensor* x) {
    check_cpu(__func__, x->device);

    int ks = this->kernel_size;
    int x_height = x->dshape[0];
    int x_width = x->dshape[1];

    DimVec res_shape = {x->shape[0], x->shape[1], x_height / ks, x_width / ks};
    int res_size = res_shape[0] * res_shape[1] * res_shape[2] * res_shape[3];
    Vec1D res_data;
    res_data.reserve(res_size);
    std::vector<int> argmaxs;
    argmaxs.reserve(res_size);

    for (int n = 0; n < x->shape[0]; n++) {
        for (int c = 0; c < x->shape[1]; c++) {
            for (int i = 0; i < x_height; i += ks) {
                int off = n * x->shape[1] * x_height * x_width + c * x_height * x_width + i * x_width;
                float max = x->data[off];
                int argmax = off;

                for (int j = 0; j < x_width; j += ks) {
                    if (i + ks <= x_height && j + ks <= x_width) {
                        for (int k0 = 0; k0 < ks; k0++) {
                            for (int k1 = 1; k1 < ks; k1++) {
                                float w = x->data[off + k0 * x_width + k1];
                                if (max < w) {
                                    max = w;
                                    argmax = off + k0 * x_width + k1;
                                }
                            }
                        }
                        argmaxs.push_back(argmax);
                        res_data.push_back(max);
                    }
                }
            }
        }
    }

    Tensor* res_tensor = new Tensor(res_shape, res_data, {x}, x->requires_grad);
    res_tensor->backward_fn = maxpool2d_backward(x, res_tensor, argmaxs);

    return res_tensor;
}

Flatten::Flatten() {}

Tensor* Flatten::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->reshape({1, (int)x->data.size()}));
    return res_tensor;
}

LeakyReLU::LeakyReLU(float negative_slope) { this->negative_slope = negative_slope; }

Tensor* LeakyReLU::forward(Tensor* x) {
    Vec1D res_data(x->data.size());
    for (std::size_t i = 0; i < x->data.size(); i++) {
        res_data[i] = leaky_relu(x->data[i], this->negative_slope);
    };

    Tensor* res_tensor = new Tensor(x->shape, res_data, {x}, x->requires_grad);
    res_tensor->backward_fn = leaky_relu_backward(x, res_tensor, this->negative_slope);

    return res_tensor;
}

ReLU::ReLU() {}

Tensor* ReLU::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->apply(relu));
    res_tensor->children = {x};
    res_tensor->backward_fn = relu_backward(x, res_tensor);

    return res_tensor;
}

Sigmoid::Sigmoid() {}

Tensor* Sigmoid::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->apply(sigmoid));
    res_tensor->children = {x};
    res_tensor->backward_fn = sigmoid_backward(x, res_tensor);

    return res_tensor;
}

Sequential::Sequential(ModuleRef layers) { this->layers = layers; }

Sequential::~Sequential() {
    for (Module* layer : this->layers) {
        delete layer;
    }
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;

    for (Module* layer : this->layers) {
        std::vector<Tensor*> lparams = layer->parameters();
        params.reserve(params.size() + distance(lparams.begin(), lparams.end()));
        params.insert(params.end(), lparams.begin(), lparams.end());
    }

    return params;
}

Tensor& Sequential::operator()(Tensor& x) {
    check_cpu(__func__, x.device);

    Tensor* lay_in = &x;
    this->layers_input.clear();
    this->layers_input.reserve(this->layers.size());

    for (std::size_t i = 0; i < this->layers.size(); i++) {
        lay_in = this->layers[i]->forward(lay_in);
        this->layers_input[i] = lay_in;
    }

    return *lay_in;
}
