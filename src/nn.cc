#include "nn.h"

#include <cmath>
#include <iostream>
#include <random>

#include "config.h"
#include "diff.h"
#include "loss.h"

float uniform_random() { return (float)rand() / RAND_MAX; }

void xavier_normal_init(Tensor* weights) {
    float fan_in = (float)weights->dshape_[0];
    float fan_out = (float)weights->dshape_[1];
    float stdev = std::sqrt(2 / (fan_in + fan_out));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd(0, stdev);

    for (Vec1D::iterator it = weights->data_.begin(), end = weights->data_.end(); it != end; it++) {
        *it = nd(gen);
    }
}

Weight::Weight(Tensor weight) {
    this->t_weight_ = new Tensor(weight.transpose());
    this->t_weight_->requires_grad_ = true;

    this->weight_ = new Tensor(weight);
    this->weight_->requires_grad_ = true;
}

Weight::~Weight() {
    delete this->weight_;
    delete this->t_weight_;
}

void Weight::operator=(Tensor& weight) {
    delete this->t_weight_;
    this->t_weight_ = new Tensor(weight.transpose());
    this->t_weight_->requires_grad_ = true;

    delete this->weight_;
    this->weight_ = new Tensor(weight);
    this->weight_->requires_grad_ = true;
}

Tensor Weight::grad(bool from_transposed) {
    return from_transposed ? this->t_weight_->grad_->transpose() : *this->weight_->grad_;
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
    this->in_features_ = in_features;
    this->out_features_ = out_features;

    Tensor weight_tensor = Tensor({out_features, in_features}, 0.0f, {}, true);
    xavier_normal_init(&weight_tensor);

    this->bias_ = new Tensor({1, out_features}, .1f, {}, true);
    this->weight_ = new Weight(weight_tensor);
}

Linear::~Linear() {
    delete this->bias_;
    delete this->weight_;
}

Tensor* Linear::forward(Tensor* x) {
    Tensor* mul_tensor = new Tensor(x->matmul((*this->weight_)()));
    Tensor* out_tensor = new Tensor(*mul_tensor + *this->bias_);
    return out_tensor;
}

std::vector<Tensor*> Linear::parameters() { return {this->weight_->t_weight_, this->bias_}; }

Conv2D::Conv2D(int in_channels, int out_channels, DimVec kernel_size, DimVec stride, DimVec padding) {
    this->in_channels_ = in_channels;
    this->out_channels_ = out_channels;
    this->stride_ = stride;
    this->padding_ = padding;

    this->padding_layer_ = new Padding(padding);

    this->bias_ = new Tensor({1, out_channels, 1, 1}, 0.1f, {}, true);

    std::vector<int> weight_shape = {out_channels, in_channels, kernel_size[0], kernel_size[1]};
    Tensor weight_tensor = Tensor(weight_shape, 0.0f, {}, true);
    xavier_normal_init(&weight_tensor);

    this->weight_ = new Weight(weight_tensor);
}

Conv2D::~Conv2D() {
    delete this->padding_layer_;
    delete this->bias_;
    delete this->weight_;
}

Tensor* Conv2D::forward(Tensor* x) {
    Tensor filter = (*this->weight_)(false);

    Tensor* padded_x = x;
    if (this->padding_[0] > 0 || this->padding_[1] > 0) {
        padded_x = this->padding_layer_->forward(x);
    }

    Tensor* cor_ten = new Tensor(padded_x->correlate(filter, this->stride_));
    cor_ten->children_ = {padded_x, this->weight_->weight_};
    cor_ten->requires_grad_ = true;
    cor_ten->backward_fn_ = correlate_backward(padded_x, this->weight_->weight_, cor_ten, this->stride_);

    return new Tensor(cor_ten->channelwise_sum(*this->bias_));
}

std::vector<Tensor*> Conv2D::parameters() { return {this->weight_->weight_, this->bias_}; }

Padding::Padding(DimVec padding, float value) {
    this->padding_ = padding;
    this->value_ = value;
}

Tensor* Padding::forward(Tensor* x) {
    Tensor padded_x = x->pad(this->padding_, this->value_);

    Tensor* res_tensor = new Tensor(padded_x.shape_, padded_x.data_, {x}, true);
    res_tensor->backward_fn_ = pad_backward(x, this->padding_, res_tensor);

    return res_tensor;
}

MaxPool2D::MaxPool2D(int kernel_size) { this->kernel_size_ = kernel_size; }

Tensor* MaxPool2D::forward(Tensor* x) {
    check_cpu(__func__, x->device_);

    int ks = this->kernel_size_;
    int x_height = x->dshape_[0];
    int x_width = x->dshape_[1];

    DimVec res_shape = {x->shape_[0], x->shape_[1], x_height / ks, x_width / ks};
    int res_size = res_shape[0] * res_shape[1] * res_shape[2] * res_shape[3];
    Vec1D res_data;
    res_data.reserve(res_size);
    std::vector<int> argmaxs;
    argmaxs.reserve(res_size);

    for (int n = 0; n < x->shape_[0]; n++) {
        for (int c = 0; c < x->shape_[1]; c++) {
            for (int i = 0; i < x_height; i += ks) {
                int off = n * x->shape_[1] * x_height * x_width + c * x_height * x_width + i * x_width;
                float max = x->data_[off];
                int argmax = off;

                for (int j = 0; j < x_width; j += ks) {
                    if (i + ks <= x_height && j + ks <= x_width) {
                        for (int k0 = 0; k0 < ks; k0++) {
                            for (int k1 = 1; k1 < ks; k1++) {
                                float w = x->data_[off + k0 * x_width + k1];
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

    Tensor* res_tensor = new Tensor(res_shape, res_data, {x}, x->requires_grad_);
    res_tensor->backward_fn_ = maxpool2d_backward(x, res_tensor, argmaxs);

    return res_tensor;
}

Flatten::Flatten() {}

Tensor* Flatten::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->reshape({1, (int)x->data_.size()}));
    return res_tensor;
}

LeakyReLU::LeakyReLU(float negative_slope) { this->negative_slope_ = negative_slope; }

Tensor* LeakyReLU::forward(Tensor* x) {
    Vec1D res_data(x->data_.size());
    for (std::size_t i = 0; i < x->data_.size(); i++) {
        res_data[i] = leaky_relu(x->data_[i], this->negative_slope_);
    };

    Tensor* res_tensor = new Tensor(x->shape_, res_data, {x}, x->requires_grad_);
    res_tensor->backward_fn_ = leaky_relu_backward(x, res_tensor, this->negative_slope_);

    return res_tensor;
}

ReLU::ReLU() {}

Tensor* ReLU::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->apply(relu));
    res_tensor->children_ = {x};
    res_tensor->backward_fn_ = relu_backward(x, res_tensor);

    return res_tensor;
}

Sigmoid::Sigmoid() {}

Tensor* Sigmoid::forward(Tensor* x) {
    Tensor* res_tensor = new Tensor(x->apply(sigmoid));
    res_tensor->children_ = {x};
    res_tensor->backward_fn_ = sigmoid_backward(x, res_tensor);

    return res_tensor;
}

Sequential::Sequential(ModuleRef layers) { this->layers_ = layers; }

Sequential::~Sequential() {
    for (Module* layer : this->layers_) {
        delete layer;
    }
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;

    for (Module* layer : this->layers_) {
        std::vector<Tensor*> lparams = layer->parameters();
        params.reserve(params.size() + distance(lparams.begin(), lparams.end()));
        params.insert(params.end(), lparams.begin(), lparams.end());
    }

    return params;
}

Tensor& Sequential::operator()(Tensor& x) {
    check_cpu(__func__, x.device_);

    Tensor* lay_in = &x;
    this->layers_input_.clear();
    this->layers_input_.reserve(this->layers_.size());

    for (std::size_t i = 0; i < this->layers_.size(); i++) {
        lay_in = this->layers_[i]->forward(lay_in);
        this->layers_input_[i] = lay_in;
    }

    return *lay_in;
}
