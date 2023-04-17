#include "tensor.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "diff.h"
#include "utility.ipp"

#include "accel/cpu.h"

#ifndef LF_NO_AVX
#include "accel/avx.h"
#endif
#ifdef LF_CUDA_AVAIL
#include "accel/cuda.cuh"
#include "cuda_runtime.h"
#endif

using namespace std::placeholders;

float EQ_TRESHOLD = 1e-4;

bool need_grad(Tensor a, Tensor b) { return a.requires_grad_ || b.requires_grad_; }

Vec1D _convert_to_vec1d(const Vec2D& data2d) {
    Vec1D data;
    for (const auto& row : data2d) {
        data.insert(data.end(), row.begin(), row.end());
    }
    return data;
}

DimVec Tensor::get_short_shape() const {
    DimVec shape = DimVec(this->shape_);
    while (shape[0] == 1 && shape.size() != 1)
        shape.erase(shape.begin());
    return shape;
}

DimVec Tensor::normalize_shape(const DimVec shape) {
    if (shape.size() > 4) throw std::runtime_error("Tensors with 5 or more dimensions are not supported");

    DimVec nshape(4, 1);
    std::copy(shape.rbegin(), shape.rend(), nshape.rbegin());
    return nshape;
}

std::size_t Tensor::size() const {
    return std::accumulate(this->shape_.begin(), this->shape_.end(), 1, std::multiplies<float>());
}

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad, Device device)
    : shape_(normalize_shape(shape)), dshape_({this->shape_[2], this->shape_[3]}), device_(device),
      requires_grad_(requires_grad), grad_(requires_grad ? new Tensor(shape, 0.0f, {}, false) : nullptr) {
    fill(0.0f);
}

Tensor::Tensor(const std::vector<int>& shape, const Vec1D& tensor, std::vector<Tensor*> children, bool requires_grad,
               Device device)
    : shape_(normalize_shape(shape)), dshape_({this->shape_[2], this->shape_[3]}), device_(device),
      requires_grad_(requires_grad), grad_(requires_grad ? new Tensor(shape, 0.0f, {}, false) : nullptr),
      children_(children) {
    fill(tensor);
}

Tensor::Tensor(const std::vector<int>& shape, const Vec2D& tensor, std::vector<Tensor*> children, bool requires_grad,
               Device device)
    : shape_(normalize_shape(shape)), dshape_({this->shape_[2], this->shape_[3]}), device_(device),
      requires_grad_(requires_grad), grad_(requires_grad ? new Tensor(shape, 0.0f, {}, false) : nullptr),
      children_(children) {
    fill(tensor);
}

Tensor::Tensor(const std::vector<int>& shape, const float constant, std::vector<Tensor*> children, bool requires_grad,
               Device device)
    : shape_(normalize_shape(shape)), dshape_({this->shape_[2], this->shape_[3]}), device_(device),
      requires_grad_(requires_grad), grad_(requires_grad ? new Tensor(shape, 0.0f, {}, false) : nullptr),
      children_(children) {
    fill(constant);
}

Tensor::~Tensor() {
    if (!this->requires_grad_ && this->grad_) {
        delete this->grad_;
    }
    this->children_.clear();
    this->children_.shrink_to_fit();
    this->data_.clear();
    this->data_.shrink_to_fit();
};

Tensor& Tensor::scalar(float value) { return *(new Tensor({1}, {value})); }

Tensor& Tensor::scalar(int value) { return *(new Tensor({1}, {(float)value})); }

Tensor Tensor::random(DimVec shape, float from, float to) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<float>());
    Vec1D res_data(size);
    std::generate(res_data.begin(), res_data.end(),
                  [=]() { return (std::rand() / static_cast<float>(RAND_MAX)) * (to - from) + std::min(from, to); });
    return Tensor(shape, res_data);
}

void Tensor::fill(const float value) { fill(Vec1D(this->size(), value)); }

void Tensor::fill(const Vec1D& data) {
    if (data.size() != this->size())
        throw std::logic_error("Wrong size of data; got: " + std::to_string(data.size()) +
                               "; expected: " + std::to_string(this->size()));

    if (this->device_ == Device::CPU) {
        this->data_ = data;
    }
    if (this->device_ == Device::CUDA) {
#ifdef LF_CUDA_AVAIL
        move_data_to_cuda(data.data(), data.size(), &this->cu_data_);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
}

void Tensor::fill(const Vec2D& data) { fill(_convert_to_vec1d(data)); }

void Tensor::add_grad(Vec1D grad) {
    if (this->size() != grad.size())
        throw std::logic_error("Wrong size of gradient; expected size: " + std::to_string(this->size()) +
                               " passed size: " + std::to_string(grad.size()));

    if (this->grad_ == nullptr) this->grad_ = new Tensor(this->shape_, 0.0f);

    auto gradib = this->grad_->data_.begin();
    std::transform(gradib, this->grad_->data_.end(), grad.begin(), gradib, std::plus<float>());
}

void Tensor::add_grad(const Tensor grad) {
    if (this->size() != grad.size())
        throw std::logic_error("Wrong size of gradient; expected size: " + std::to_string(this->size()) +
                               " passed size: " + std::to_string(grad.size()));

    if (this->grad_ == nullptr) this->grad_ = new Tensor(this->shape_, 0.0f);
    if (this->device_ == Device::CPU) {
        auto gradib = this->grad_->data_.begin();
        std::transform(gradib, this->grad_->data_.end(), grad.data_.begin(), gradib, std::plus<float>());
    } else {
        add_cuda(this->grad_->cu_data_, grad.cu_data_, this->grad_->cu_data_, this->grad_->size(), grad.size());
    }
}

void Tensor::set_grad(Vec1D grad) {
    if (this->grad_ == nullptr) this->grad_ = new Tensor(this->shape_);
    this->grad_->fill(grad);
}

void Tensor::check_same_shape(Tensor& other, bool allow_scalar) const {
    std::vector<int> tshape = this->shape_;
    std::vector<int> oshape = other.shape_;

    if (tshape != oshape) {
        std::string lstr = vector_to_string(tshape);
        std::string rstr = vector_to_string(oshape);

        throw std::logic_error("Tensors doesn't have same shape; lval" + lstr + ";rval:" + rstr);
    }
}

Tensor Tensor::apply(std::function<float(float)> function) {
    Vec1D res_data(this->data_.size());
    std::transform(this->data_.begin(), this->data_.end(), res_data.begin(), function);
    return Tensor(this->shape_, res_data, {}, this->requires_grad_);
}

Tensor Tensor::apply_operator(Tensor& other, OperationFunc operation_fn) {
    DimVec res_shape = this->shape_;

    DimVec t_short_shape = this->get_short_shape();
    DimVec o_short_shape = other.get_short_shape();

    auto tdb = this->data_.begin();
    auto tde = this->data_.end();
    auto odb = other.data_.begin();

    // adjust data if one of tensors is scalar
    Vec1D t_adj_tensor;
    Vec1D o_adj_tensor;
    if (t_short_shape.size() == 1 && t_short_shape[0] == 1) {
        t_adj_tensor = Vec1D(other.data_.size(), this->data_[0]);
        tdb = t_adj_tensor.begin();
        tde = t_adj_tensor.end();
        res_shape = other.shape_;
    } else if (o_short_shape.size() == 1 && o_short_shape[0] == 1) {
        o_adj_tensor = Vec1D(this->data_.size(), other.data_[0]);
        odb = o_adj_tensor.begin();
    }

    Vec1D res_vec;
    std::transform(tdb, tde, odb, std::back_inserter(res_vec), operation_fn);
    return Tensor(res_shape, res_vec, {this, &other}, need_grad(*this, other));
}

Tensor Tensor::operator+(float value) {
    Vec1D nd = this->data_;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c + value; });
    return Tensor(this->shape_, nd);
}

Tensor Tensor::operator+(Tensor& other) {
    Tensor out = (this->device_ == Device::CUDA) ? Tensor(this->shape_, 0.0f, {this, &other}, need_grad(*this, other))
                                                 : apply_operator(other, add);
    if (this->device_ == Device::CUDA)
        add_cuda(this->cu_data_, other.cu_data_, out.cu_data_, this->size(), other.size());
    if (out.requires_grad_) out.backward_fn_ = add_backward(this, &other, &out);
    return out;
}

Tensor Tensor::operator-(float value) {
    Vec1D nd = this->data_;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c - value; });
    return Tensor(this->shape_, nd);
}

Tensor Tensor::operator-(Tensor& other) {
    Tensor out = (this->device_ == Device::CUDA) ? Tensor(this->shape_, 0.0f, {this, &other}, need_grad(*this, other))
                                                 : apply_operator(other, sub);
    if (this->device_ == Device::CUDA)
        sub_cuda(this->cu_data_, other.cu_data_, out.cu_data_, this->size(), other.size());
    if (out.requires_grad_) out.backward_fn_ = sub_backward(this, &other, &out);
    return out;
}

Tensor Tensor::operator*(float value) {
    Vec1D nd = this->data_;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c * value; });
    return Tensor(this->shape_, nd);
}

Tensor Tensor::operator*(Tensor& other) {
    Tensor out = (this->device_ == Device::CUDA) ? Tensor(this->shape_, 0.0f, {this, &other}, need_grad(*this, other))
                                                 : apply_operator(other, mul);
    if (this->device_ == Device::CUDA)
        mul_cuda(this->cu_data_, other.cu_data_, out.cu_data_, this->size(), other.size());
    if (out.requires_grad_) out.backward_fn_ = mul_backward(this, &other, &out);
    return out;
}

Tensor Tensor::operator/(float value) {
    Vec1D nd = this->data_;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c / value; });
    return Tensor(this->shape_, nd);
}

Tensor Tensor::operator/(Tensor& other) {
    Tensor out = (this->device_ == Device::CUDA) ? Tensor(this->shape_, 0.0f, {this, &other}, need_grad(*this, other))
                                                 : apply_operator(other, ddiv);
    if (this->device_ == Device::CUDA)
        div_cuda(this->cu_data_, other.cu_data_, out.cu_data_, this->size(), other.size());
    if (out.requires_grad_) out.backward_fn_ = ddiv_backward(this, &other, &out);
    return out;
}

Tensor Tensor::pow(float exp) {
    Tensor* out = nullptr;
    if (this->device_ == Device::CPU) {
        Vec1D nd = this->data_;
        std::transform(nd.begin(), nd.end(), nd.begin(), [exp](float& c) { return std::pow(c, exp); });
        out = new Tensor(this->shape_, nd);
    } else {
        out = new Tensor(this->shape_, 0.0f);
        pow_const_cuda(this->cu_data_, exp, out->cu_data_, this->size());
    }
    return *out;
}

Tensor Tensor::pow(Tensor& exp) {
    Tensor* out = nullptr;
    if (this->device_ == Device::CPU) {
        int iexp = exp.data_[0];
        Vec1D nd = this->data_;
        std::transform(nd.begin(), nd.end(), nd.begin(), [iexp](float& c) { return std::pow(c, iexp); });
        out = new Tensor(this->shape_, nd, {this, &exp}, exp.requires_grad_);
    } else {
        out = new Tensor(this->shape_, 0.0f, {this, &exp}, exp.requires_grad_);
        pow_cuda(this->cu_data_, exp.cu_data_, out->cu_data_, this->size());
    }
    if (out->requires_grad_) out->backward_fn_ = pow_backward(this, &exp, out);
    return *out;
}

bool Tensor::has_same_shape(Tensor& other) {
    return are_same_vectors(this->get_short_shape(), other.get_short_shape());
}

void Tensor::operator+=(Tensor other) {
    check_same_shape(other, false);
    std::transform(this->data_.begin(), this->data_.end(), other.data_.begin(), this->data_.begin(),
                   std::plus<float>());
}

void Tensor::operator-=(Tensor other) {
    check_same_shape(other, false);
    std::transform(this->data_.begin(), this->data_.end(), other.data_.begin(), this->data_.begin(),
                   std::minus<float>());
}

bool Tensor::operator==(Tensor& other) {
    if (!has_same_shape(other) || this->device_ != other.device_) return false;

    if (this->device_ == Device::CUDA) {
#ifdef LF_CUDA_AVAIL
        return compare_arrays_cuda(this->cu_data_, other.cu_data_, EQ_TRESHOLD, this->size());
#else
        throw std::runtime_error("CUDA not available");
#endif
    }

    for (std::size_t i = 0; i < this->data_.size(); i++) {
        if (fabs(this->data_[i] - other.data_[i]) > EQ_TRESHOLD) return false;
    }
    return true;
}

int Tensor::argmax() {
    int maxi = 0;
    float max = this->data_[0];

    for (std::size_t i = 1; i < this->data_.size(); i++) {
        if (this->data_[i] > max) {
            max = this->data_[i];
            maxi = i;
        }
    }

    return maxi;
}

Tensor Tensor::sqrt() {
    Vec1D nd(this->data_);
    std::transform(nd.begin(), nd.end(), nd.begin(), (float (*)(float))std::sqrt);
    return Tensor(this->shape_, nd);
}

Tensor Tensor::log() {
    Tensor* out = nullptr;
    if (this->device_ == Device::CPU) {
        out = new Tensor(this->shape_, this->data_);
        std::transform(out->data_.begin(), out->data_.end(), out->data_.begin(), (float (*)(float))std::log);
    } else {
        out = new Tensor(this->shape_, 0.0f);
        log_cuda(this->cu_data_, out->cu_data_, this->size());
    }
    return *out;
}

float Tensor::max() { return *std::max_element(std::begin(this->data_), std::end(this->data_)); }

float Tensor::min() { return *std::min_element(std::begin(this->data_), std::end(this->data_)); }

float Tensor::sum() { return std::reduce(this->data_.begin(), this->data_.end()); }

Tensor Tensor::matmul(Tensor& other) {
    if (this->dshape_[1] != other.dshape_[0]) {
        throw std::logic_error("matmul error: wrong shape of tensors; " + std::to_string(this->dshape_[1]) + " and " +
                               std::to_string(other.dshape_[0]));
    }
    if (this->shape_[0] != other.shape_[0] || this->shape_[1] != other.shape_[1]) {
        throw std::logic_error("matmul error: tensors must have same outter dimensions");
    }

    DimVec res_dim = {this->shape_[0], this->shape_[1], this->dshape_[0], other.dshape_[1]};
    Tensor res_tensor = Tensor(res_dim, 0.0f, {this, &other}, need_grad(*this, other));

    if (this->device_ == Device::CPU) {
        MatmulFunc mm_fn;
#ifdef LF_NO_AVX
        mm_fn = std::bind(matmul_cpu, _1, _2, _3, _4, _5, _6);
#else
        mm_fn = std::bind(matmul_avx, _1, _2, _3, _4, _5, _6);
#endif
        if (this->shape_[0] > 1 || this->shape_[1] > 1)
            _matmul_deep_cpu(*this, other, res_tensor.data_.data(), mm_fn);
        else
            mm_fn(this->data_.data(), other.data_.data(), res_tensor.data_.data(), this->dshape_[0], this->dshape_[1],
                  other.dshape_[1]);
    } else if (this->device_ == Device::CUDA) {
#ifdef LF_CUDA_AVAIL
        if (this->shape_[0] > 1 || this->shape_[1] > 1)
            matmul_deep_cuda(this->cu_data_, other.cu_data_, res_tensor.cu_data_, this->dshape_[0], this->dshape_[1],
                             other.dshape_[1], this->shape_[1], this->shape_[0]);
        else
            matmul_cuda(this->cu_data_, other.cu_data_, res_tensor.cu_data_, this->dshape_[0], this->dshape_[1],
                        other.dshape_[1]);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }

    if (res_tensor.requires_grad_) res_tensor.backward_fn_ = matmul_backward(this, &other, &res_tensor);

    return res_tensor;
}

Tensor Tensor::channelwise_sum(Tensor& other) {
    if (this->shape_[1] != static_cast<int>(other.data_.size()))
        throw std::logic_error("channelwise_sum error: wrong size of tensor");

    Tensor res_ten = Tensor(this->shape_, this->data_, {this, &other}, need_grad(*this, other));

    int res_wh = res_ten.shape_[2] * res_ten.shape_[3];
    for (int n = 0; n < res_ten.shape_[0]; n++) {
        for (int c = 0; c < res_ten.shape_[1]; c++) {
            auto resb = res_ten.data_.begin() + n * res_ten.shape_[1] * res_wh + c * res_wh;
            float oc = other.data_[c];
            std::transform(resb, resb + res_wh, resb, [oc](float& res) { return res + oc; });
        }
    }

    if (res_ten.requires_grad_) res_ten.backward_fn_ = channelwise_sum_backward(this, &other, &res_ten);
    return res_ten;
}

Tensor Tensor::reshape(DimVec new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<float>());
    if (new_size != this->size()) throw std::logic_error("reshape error: sizes of tensors doesn't match");

    Tensor res_tensor = Tensor(new_shape, this->data_, {this}, this->requires_grad_);
    if (res_tensor.requires_grad_) res_tensor.backward_fn_ = reshape_backward(this, &res_tensor);
    return res_tensor;
}

Tensor Tensor::transpose() {
    if (this->device_ == Device::CUDA) {
        Tensor out = Tensor({this->shape_[0], this->shape_[1], this->dshape_[1], this->dshape_[0]}, 0.0f);
        transpose_cuda(this->cu_data_, out.cu_data_, this->shape_[0], this->shape_[1], this->shape_[2],
                       this->shape_[3]);
        return out;
    } else {
        return transpose_cpu(*this);
    }
}

Tensor Tensor::get_block(int n) {
    if (this->shape_[0] <= n) throw std::logic_error("get_channel: Wrong index of channel\n");

    int block_size = this->shape_[1] * this->dshape_[0] * this->dshape_[1];
    int block_start = n * block_size;
    Vec1D block_data(this->data_.begin() + block_start, this->data_.begin() + block_start + block_size);
    return Tensor({this->shape_[1], this->dshape_[0], this->dshape_[1]}, block_data, {this}, this->requires_grad_);
}

Tensor Tensor::get_channel(int channel) {
    if (this->shape_[0] != 1) throw std::logic_error("get_channel: Only tensors with batch_size == 1 supported\n");
    if (this->shape_[1] <= channel) throw std::logic_error("get_channel: Wrong index of channel\n");

    int channel_start = this->dshape_[0] * this->dshape_[1] * channel;
    Tensor res = Tensor({1, 1, this->dshape_[0], this->dshape_[1]});
    std::transform(res.data_.begin(), res.data_.end(), this->data_.begin() + channel_start, res.data_.begin(),
                   std::plus<float>());
    return res;
}

void Tensor::add_channel(const Tensor& channel) {
    this->shape_[1]++;
    this->data_.reserve(this->size());
    for (std::size_t i = 0; i < channel.size(); i++) {
        this->data_.push_back(channel.data_[i]);
    }
}

Tensor Tensor::correlate(Tensor& filter, DimVec stride, DimVec padding) {
    check_cpu(__func__, this->device_);

    if (this->shape_[1] != filter.shape_[1]) {
        throw std::logic_error("correlate: Wrong shape of tensors at correlation\n x: " + this->to_string() +
                               "\n and \n filter: " + filter.to_string() + "\n");
    }
    if (stride.size() != 2) throw std::logic_error("correlate: stride should have size 2");
    if (padding.size() != 2) throw std::logic_error("correlate: padding should have size 2");

    CorrelateFunc cor_fn = std::bind(correlate_cpu, _1, _2, _3, _4);
    if (this->device_ == Device::CUDA) {
#ifdef LF_CUDA_AVAIL
        throw std::logic_error("cuda correlation not yet implemented");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
#ifndef LF_NO_AVX
        cor_fn = std::bind(correlate_avx, _1, _2, _3, _4);
#endif
    }

    return cor_fn(*this, filter, stride, padding);
}

Tensor Tensor::pad(DimVec padding, float value) {
    check_cpu(__func__, this->device_);

    int hpad = padding[0];
    int wpad = padding[1];

    Tensor res_tensor =
        Tensor({this->shape_[0], this->shape_[1], this->shape_[2] + hpad * 2, this->shape_[3] + wpad * 2}, value);

#pragma omp parallel for
    for (int n = 0; n < res_tensor.shape_[0]; n++) {
        int ni = n * res_tensor.shape_[1] * res_tensor.shape_[2] * res_tensor.shape_[3];
        int tni = n * this->shape_[1] * this->shape_[2] * this->shape_[3];
        for (int c = 0; c < res_tensor.shape_[1]; c++) {
            int ci = c * res_tensor.shape_[2] * res_tensor.shape_[3];
            int tci = c * this->shape_[2] * this->shape_[3];
            for (int h = 0; h < res_tensor.shape_[2]; h++) {
                int hi = h * res_tensor.shape_[3];
                int thi = (h - hpad) * this->shape_[3];
                for (int w = 0; w < res_tensor.shape_[3]; w++) {
                    if (h < hpad || h >= this->shape_[2] + hpad || w < wpad || w >= this->shape_[3] + wpad) continue;

                    res_tensor.data_[ni + ci + hi + w] = this->data_[tni + tci + thi + w - wpad];
                }
            }
        }
    }

    return res_tensor;
}

Tensor Tensor::rot180() {
    check_cpu(__func__, this->device_);

    int m = this->dshape_[0];
    int n = this->dshape_[1];

    Tensor res_ten = *this;

#pragma omp parallel for
    for (int bs = 0; bs < this->shape_[0] * this->shape_[1]; bs++) {
        int off = bs * m * n;
        for (int k = 0; k < m * n / 2; k++) {
            int r = k / n;
            int c = k % n;
            std::swap(res_ten.data_[off + r * n + c], res_ten.data_[off + (m - r - 1) * n + (n - c - 1)]);
        }
    }

    return res_ten;
}

Tensor Tensor::to(Device device) {
    if (this->device_ == Device::CPU) {
        return Tensor(this->shape_, this->data_, this->children_, this->requires_grad_, device);
    } else {
#ifdef LF_CUDA_AVAIL
        std::vector<float> host_data(this->size());
        // move_data_to_host(&host_data[0], this->cu_data_, host_data.size());
        return Tensor(this->shape_, host_data, this->children_, this->requires_grad_, device);
#else
        throw std::runtime_error("CUDA not available");
#endif
    }
}

void Tensor::backward() {
    if (this->backward_fn_ != nullptr) this->backward_fn_();

    for (std::vector<Tensor*>::iterator it = this->children_.begin(); it != this->children_.end(); it++)
        (*it)->backward();
}

std::string Tensor::to_string() {
    std::string res = "tensor((";
    std::string dev = this->device_ == Device::CPU ? "cpu" : "cuda";
    std::string data_str = "[";

    float* tmp_data = &this->data_[0];
    if (this->device_ == Device::CUDA) {
        tmp_data = (float*)malloc(this->size() * sizeof(float));
        move_data_to_host(tmp_data, this->cu_data_, this->size());
    }

    for (size_t i = 0; i < this->size(); i++) {
        data_str += std::to_string(tmp_data[i]);
        data_str += (i == this->size() - 1) ? "]" : ", ";

        if (i == 20) {
            data_str += "... ]";
            break;
        }
    }

    for (int shape : this->shape_) {
        res += std::to_string(shape) + ",";
    }
    res.pop_back();

    return res + "), " + dev + ") " + data_str;
}
