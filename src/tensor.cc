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

using namespace std::placeholders;

float EQ_TRESHOLD = 1e-4;

bool need_grad(Tensor a, Tensor b) { return a.require_grad || b.require_grad; }

DimVec Tensor::get_short_shape() {
    DimVec shape = DimVec(this->shape);

    while (shape[0] == 1 && shape.size() != 1) {
        shape.erase(shape.begin());
    }
    return shape;
};

DimVec Tensor::normalize_shape(DimVec shape) {
    if (shape.size() > 4) {
        throw std::runtime_error("5 dimensional or higher tensor are not supported");
    }
    DimVec nshape = DimVec(4 - shape.size(), 1);
    for (std::size_t i = 0; i < shape.size(); i++) {
        nshape.push_back(shape[i]);
    }

    return nshape;
}

std::size_t Tensor::size() {
    return std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<float>());
}

Tensor::Tensor(const std::vector<int>& shape, bool require_grad) {
    this->shape = normalize_shape(shape);
    this->dshape = {this->shape[2], this->shape[3]};
    this->dim = get_short_shape().size();

    this->require_grad = require_grad;
    if (require_grad) {
        this->grad = new Tensor(shape, 0.0f, {}, false);
    }
}

Tensor::Tensor(const std::vector<int>& shape, const Vec1D tensor, std::vector<Tensor*> children, bool require_grad) {
    this->shape = normalize_shape(shape);
    this->dshape = {this->shape[2], this->shape[3]};
    this->dim = get_short_shape().size();

    this->require_grad = require_grad;
    if (require_grad) {
        this->grad = new Tensor(shape, 0.0f, {}, false);
        this->children = children;
    }

    fill(tensor);
}

Tensor::Tensor(const std::vector<int>& shape, const Vec2D tensor, std::vector<Tensor*> children, bool require_grad) {
    this->shape = normalize_shape(shape);
    this->dshape = {this->shape[2], this->shape[3]};
    this->dim = get_short_shape().size();

    this->require_grad = require_grad;
    if (require_grad) {
        this->grad = new Tensor(shape, 0.0f, {}, false);
        this->children = children;
    }

    fill(tensor);
}

Tensor::Tensor(const std::vector<int>& shape, const float constant, std::vector<Tensor*> children, bool require_grad) {
    this->shape = normalize_shape(shape);
    this->dshape = {this->shape[2], this->shape[3]};
    this->dim = get_short_shape().size();

    this->require_grad = require_grad;
    if (require_grad) {
        this->grad = new Tensor(shape, 0.0f, {}, false);
        this->children = children;
    }

    this->data = Vec1D(this->size(), constant);
}

Tensor::~Tensor() {
    if (!this->require_grad && this->grad) {
        delete this->grad;
    }
    this->data.clear();
};

Tensor Tensor::scalar(float value) { return Tensor({1}, value); }

Tensor Tensor::scalar(int value) { return Tensor({1}, (float)value); }

Tensor Tensor::random(DimVec shape, float from, float to) {
    Tensor rand_tensor = Tensor(shape);

    for (std::size_t i = 0; i < rand_tensor.size(); i++) {
        float rand = (std::rand() / RAND_MAX) * (to - from) + std::abs(from);
        rand_tensor.data.push_back(rand);
    }

    return rand_tensor;
}

void Tensor::fill(float value) { std::fill(this->data.begin(), this->data.end(), value); }

void Tensor::fill(Vec1D data) {
    if (data.size() != this->size()) {
        std::string message =
            "Wrong size of data; got: " + std::to_string(data.size()) + "; expected: " + std::to_string(this->size());
        throw std::logic_error(message);
    }
    this->data = data;
}

void Tensor::fill(Vec2D data) {
    std::size_t len = 0;
    for (std::size_t i = 0; i < data.size(); i++) {
        for (std::size_t j = 0; j < data[0].size(); j++) {
            this->data.push_back(data[i][j]);
            len++;
        }
    }

    if (len != this->size()) {
        throw std::logic_error("Wrong size of data");
    }
}

void Tensor::add_grad(Vec1D grad) {
    if (this->size() != grad.size()) {
        throw std::logic_error("Wrong size of gradient; expected size: " + std::to_string(this->size()) +
                               " passed size: " + std::to_string(grad.size()));
    }

    if (this->grad == nullptr) {
        this->grad = new Tensor(this->shape, 0.0f);
    }

    Vec1D::iterator gradib = this->grad->data.begin();
    Vec1D::iterator gradie = this->grad->data.end();
    std::transform(gradib, gradie, grad.begin(), gradib, std::plus<float>());
}

void Tensor::set_grad(Vec1D grad) {
    if (this->grad == nullptr) {
        this->grad = new Tensor(this->shape);
    }
    this->grad->fill(grad);
}

void Tensor::check_same_shape(Tensor& other, bool allow_scalar) {
    std::vector<int> tshape = this->shape;
    std::vector<int> oshape = other.shape;

    if (tshape != oshape) {
        std::string lstr = vector_to_string(tshape);
        std::string rstr = vector_to_string(oshape);

        throw std::logic_error("Tensors doesn't have same shape; lval" + lstr + ";rval:" + rstr);
    }
}

Tensor Tensor::apply(std::function<float(float)> function) {
    Vec1D res_data(this->data.size());
    std::transform(data.begin(), data.end(), res_data.begin(), function);

    return Tensor(this->shape, res_data, {}, this->require_grad);
}

Tensor Tensor::apply_operator(Tensor& other, OperationFc operation_fn) {
    DimVec res_shape = this->shape;

    DimVec t_short_shape = this->get_short_shape();
    DimVec o_short_shape = other.get_short_shape();
    bool t_need_adjust = t_short_shape.size() == 1 && t_short_shape[0] == 1;
    bool o_need_adjust = o_short_shape.size() == 1 && o_short_shape[0] == 1;

    if (!t_need_adjust && !o_need_adjust) {
        int data_size = this->data.size();

        Vec1D res_vec(data_size);
        for (int i = 0; i < data_size; i++) {
            res_vec[i] = operation_fn(this->data[i], other.data[i]);
        }

        return Tensor(res_shape, res_vec, {this, &other}, need_grad(*this, other));
    }

    Vec1D t_adj_tensor = this->data;
    Vec1D o_adj_tensor = other.data;

    // adjust data if one of tensors is scalar
    if (t_short_shape.size() == 1 && t_short_shape[0] == 1) {
        t_adj_tensor = Vec1D(other.data.size(), this->data[0]);
        res_shape = other.shape;
    } else if (o_short_shape.size() == 1 && o_short_shape[0] == 1) {
        o_adj_tensor = Vec1D(this->data.size(), other.data[0]);
    }

    Vec1D res_vec(t_adj_tensor.size());
    for (std::size_t i = 0; i < t_adj_tensor.size(); i++) {
        res_vec[i] = operation_fn(t_adj_tensor[i], o_adj_tensor[i]);
    }

    return Tensor(res_shape, res_vec, {this, &other}, need_grad(*this, other));
}

Tensor Tensor::operator+(float value) {
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c + value; });

    return Tensor(this->shape, nd);
}

Tensor Tensor::operator+(Tensor& other) {
    Tensor out = apply_operator(other, add);

    if (out.require_grad) {
        out.backward_fn = add_backward(this, &other, &out);
    }

    return out;
}

Tensor Tensor::operator-(float value) {
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c - value; });

    return Tensor(this->shape, nd);
}

Tensor Tensor::operator-(Tensor& other) {
    Tensor out = apply_operator(other, sub);

    if (out.require_grad) {
        out.backward_fn = sub_backward(this, &other, &out);
    }

    return out;
}

Tensor Tensor::operator*(float value) {
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c * value; });

    return Tensor(this->shape, nd);
}

Tensor Tensor::operator*(Tensor& other) {
    Tensor out = apply_operator(other, mul);
    if (out.require_grad) {
        out.backward_fn = mul_backward(this, &other, &out);
    }

    return out;
}

Tensor Tensor::operator/(float value) {
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [value](float& c) { return c / value; });

    return Tensor(this->shape, nd);
}

Tensor Tensor::operator/(Tensor& other) {
    Tensor out = apply_operator(other, ddiv);
    if (out.require_grad) {
        out.backward_fn = ddiv_backward(this, &other, &out);
    }

    return out;
}

Tensor Tensor::pow(float exp) {
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [exp](float& c) { return std::pow(c, exp); });
    return Tensor(this->shape, nd);
}

Tensor Tensor::pow(Tensor& exp) {
    int iexp = exp.data[0];
    Vec1D nd = this->data;
    std::transform(nd.begin(), nd.end(), nd.begin(), [iexp](float& c) { return std::pow(c, iexp); });

    Tensor out = Tensor(this->shape, nd, {this, &exp}, exp.require_grad);
    if (out.require_grad) {
        out.backward_fn = pow_backward(this, &exp, &out);
    }

    return out;
}

bool Tensor::has_same_shape(Tensor& other) {
    return are_same_vectors(this->get_short_shape(), other.get_short_shape());
}

void Tensor::operator+=(Tensor other) {
    check_same_shape(other, false);
    std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::plus<float>());
}

void Tensor::operator-=(Tensor other) {
    check_same_shape(other, false);
    std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::minus<float>());
}

bool Tensor::operator==(Tensor& other) {
    if (!has_same_shape(other)) {
        return false;
    }

    for (std::size_t i = 0; i < this->data.size(); i++) {
        if (fabs(this->data[i] - other.data[i]) > EQ_TRESHOLD) {
            return false;
        }
    }
    return true;
}

Vec1D Tensor::get_col(int col_num) {
    Vec1D col;
    for (std::size_t i = col_num; i < data.size(); i += this->dshape[1]) {
        col.push_back(data[i]);
    }
    return col;
}

Vec1D Tensor::get_row(int row_num) {
    Vec1D::const_iterator startit = this->data.begin() + row_num * this->dshape[1];
    return Vec1D(startit, startit + this->dshape[1]);
}

int Tensor::argmax() {
    int maxi = 0;
    float max = this->data[0];

    for (std::size_t i = 1; i < this->data.size(); i++) {
        if (this->data[i] > max) {
            max = this->data[i];
            maxi = i;
        }
    }

    return maxi;
}

Tensor Tensor::sqrt() {
    Vec1D nd(this->data);
    std::transform(nd.begin(), nd.end(), nd.begin(), (float (*)(float))std::sqrt);
    return Tensor(this->shape, nd);
}

float Tensor::max() { return *std::max_element(std::begin(this->data), std::end(this->data)); }

float Tensor::min() { return *std::min_element(std::begin(this->data), std::end(this->data)); }

float Tensor::sum() { return std::reduce(this->data.begin(), this->data.end()); }

#ifdef AVX
#include <immintrin.h>

void Tensor::_matmul(Tensor& other, float* res, int tof, int oof) {
    int theight = this->dshape[0];
    int owidth = other.dshape[1];
    int twidth = this->dshape[1];

    const int BSB = 64;
    const int BSA = 4;

    for (int bb = 0; bb < theight; bb += BSB) {
        float bbm = std::min(bb + BSB, theight);
        for (int ba = 0; ba < twidth; ba += BSA) {
            float bam = std::min(ba + BSA, twidth);
            for (int i = bb; i < bbm; i++) {
                for (int j = ba; j < bam; j++) {
                    __m256 vec_a = _mm256_set1_ps(this->data[i * twidth + j]);

                    int k;
                    for (k = 0; k <= owidth - 8; k += 8) {
                        _mm256_storeu_ps(&res[i * owidth + k],
                                         _mm256_fmadd_ps(vec_a, _mm256_loadu_ps(&other.data[j * owidth + k]),
                                                         _mm256_loadu_ps(&res[i * owidth + k])));
                    }

                    // compute exceding elements
                    for (int q = 0; q + k < owidth; q++) {
                        res[i * owidth + k + q] +=
                            this->data[tof + i * twidth + j] * other.data[oof + j * owidth + k + q];
                    }
                }
            }
        }
    }
}
#else

void Tensor::_matmul(Tensor& other, float* res, int tof, int oof) {
    int theight = this->dshape[0];
    int owidth = other.dshape[1];
    int twidth = this->dshape[1];

    int bs = (twidth < 64) ? twidth : (twidth < 256) ? 64 : 256;
    int bss = (owidth < 64) ? owidth : (owidth < 256) ? 64 : 256;

    for (int ba = 0; ba < twidth; ba += bs) {
        for (int bb = 0; bb < owidth; bb += bss) {
            for (int i = 0; i < theight; i++) {
                for (int k = ba; k < std::min(ba + bs, twidth); k++) {
                    for (int j = bb; j < std::min(bb + bss, owidth); j++) {
                        res[i * owidth + j] += this->data[tof + i * twidth + k] * other.data[oof + k * owidth + j];
                    }
                }
            }
        }
    }
}

#endif

void Tensor::_matmul_deep(Tensor& other, float* res, std::function<void(Tensor&, float*, int, int)> mm) {
    int theight = this->dshape[0];
    int owidth = other.dshape[1];
    int twidth = this->dshape[1];
    int oheight = other.dshape[0];

    int i = 0;
    for (int b0 = 0; b0 < this->shape[0]; b0++) {
        for (int b1 = 0; b1 < this->shape[1]; b1++) {
            int tof = b0 * this->shape[1] * theight * twidth + b1 * theight * twidth;
            int oof = b0 * other.shape[1] * owidth * oheight + b1 * owidth * oheight;

            mm(other, &res[i], tof, oof);
            i += theight * owidth;
        }
    }
}

Tensor Tensor::matmul(Tensor& other) {
    if (this->dshape[1] != other.dshape[0]) {
        throw std::logic_error("matmul error: wrong shape of tensors; " + std::to_string(this->dshape[1]) + " and " +
                               std::to_string(other.dshape[0]));
    }
    if (this->shape[0] != other.shape[0] || this->shape[1] != other.shape[1]) {
        throw std::logic_error("matmul error: tensors must have same outter dimensions");
    }

    DimVec res_dim = {this->shape[0], this->shape[1], this->dshape[0], other.dshape[1]};
    Vec1D res_data(this->shape[0] * this->shape[1] * this->dshape[0] * other.dshape[1], 0.0f);

    std::function<void(Tensor&, float*, int, int)> mm = std::bind(&Tensor::_matmul, this, _1, _2, _3, _4);

    (this->shape[0] > 1 || this->shape[1] > 1) ? _matmul_deep(other, res_data.data(), mm)
                                               : mm(other, res_data.data(), 0, 0);

    Tensor res_tensor = Tensor(res_dim, res_data, {this, &other}, need_grad(*this, other));
    if (res_tensor.require_grad) {
        res_tensor.backward_fn = matmul_backward(this, &other, &res_tensor);
    }

    return res_tensor;
}

Tensor Tensor::channelwise_sum(Tensor& other) {
    if (this->shape[1] != (int)other.data.size()) {
        throw std::logic_error("channelwise_sum error: wrong size of tensor");
    }

    Tensor res_ten = Tensor(this->shape, this->data, {this, &other}, need_grad(*this, other));

    int res_wh = res_ten.shape[2] * res_ten.shape[3];
    for (int n = 0; n < res_ten.shape[0]; n++) {
        for (int c = 0; c < res_ten.shape[1]; c++) {
            int off = n * res_ten.shape[1] * res_wh + c * res_wh;

            for (int i = 0; i < res_wh; i++) {
                res_ten.data[off + i] += other.data[c];
            }
        }
    }

    if (res_ten.require_grad) {
        res_ten.backward_fn = channelwise_sum_backward(this, &other, &res_ten);
    }

    return res_ten;
}

Tensor Tensor::reshape(DimVec new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<float>());

    if (new_size != this->size()) {
        throw std::logic_error("reshape error: sizes of tensors doesn't match");
    }

    Tensor res_tensor = Tensor(new_shape, this->data, {this}, this->require_grad);
    if (res_tensor.require_grad) {
        res_tensor.backward_fn = reshape_backward(this, &res_tensor);
    }

    return res_tensor;
}

Tensor Tensor::transpose() {
    DimVec trans_shape = {this->shape[0], this->shape[1], this->dshape[1], this->dshape[0]};

    Vec1D trans_data(this->data.size());
    int theight = this->dshape[1];
    int twidth = this->dshape[0];

    for (int n = 0; n < this->shape[0]; n++) {
        int boh = n * this->shape[1] * theight * twidth;
        for (int c = 0; c < this->shape[1]; c++) {
            int tof = boh + c * theight * twidth;
            for (int i = 0; i < theight; i++) {
                for (int j = 0; j < twidth; j++) {
                    trans_data[tof + i * twidth + j] = this->data[tof + j * theight + i];
                }
            }
        }
    }

    return Tensor(trans_shape, trans_data);
}

Tensor Tensor::get_block(int n) {
    if (this->shape[0] <= n) {
        throw std::logic_error("get_channel: Wrong index of channel\n");
    }

    int bs = this->shape[1] * this->dshape[0] * this->dshape[1];
    int si = n * bs;
    int ei = (n + 1) * bs;

    DimVec block_shape = this->shape;
    block_shape[0] = 1;

    Vec1D block_data(ei - si);

    for (int i = si; i < ei; i++) {
        block_data[i - si] = this->data[i];
    }

    return Tensor(block_shape, block_data);
}

Tensor Tensor::get_channel(int channel) {
    if (this->shape[1] <= channel) {
        throw std::logic_error("get_channel: Wrong index of channel\n");
    }

    int si = this->dshape[0] * this->dshape[1] * channel;
    int ei = this->dshape[0] * this->dshape[1] * (channel + 1);

    DimVec channel_shape = this->shape;
    channel_shape[1] = 1;

    Vec1D channel_data(ei - si);

    for (int i = si; i < ei; i++) {
        channel_data[i - si] = this->data[i];
    }

    return Tensor(channel_shape, channel_data);
}

Tensor Tensor::add_channel(Tensor& channel) {
    if (this->shape[0] != channel.shape[0] && this->dshape[1] != channel.dshape[1] &&
        this->shape[2] != channel.shape[2]) {
        throw std::logic_error("add_channel: Wrong size of new channel\n");
    }
    int dsize = this->dshape[0] * this->dshape[1];
    int bs = this->shape[1] * dsize;

    DimVec res_shape = this->shape;
    res_shape[1]++;

    Vec1D res_data(res_shape[0] * res_shape[1] * dsize);
    int li = 0;

    for (int n = 0; n < this->shape[0]; n++) {
        int si = n * bs;
        int ei = (n + 1) * bs;

        for (int i = si; i < ei; i++) {
            res_data[i - si] = this->data[i];
        }

        for (; li < dsize; li++) {
            res_data[ei + li] = channel.data[li];
        }
    }

    return Tensor(res_shape, res_data);
}

Tensor Tensor::correlate(Tensor& filter, DimVec stride, DimVec padding) {
    if (this->shape[1] != filter.shape[1]) {
        throw std::logic_error("correlate: Wrong shape of tensors at correlation\n x: " + this->to_string() +
                               "\n and \n filter: " + filter.to_string() + "\n");
    }

    int fil_height = filter.dshape[0];
    int fil_width = filter.dshape[1];
    int x_height = this->dshape[0];
    int x_width = this->dshape[1];

    int nrows = floor((x_height - fil_height + 2 * padding[0]) / stride[0] + 1);
    int ncols = floor((x_width - fil_width + 2 * padding[1]) / stride[1] + 1);

    int fil_size = fil_height * fil_width;
    int x_size = x_height * x_width;

    Vec1D res_data(filter.shape[0] * nrows * ncols, 0.0f);

    for (int n = 0; n < filter.shape[0]; n++) {
        int rof = n * nrows * ncols;
        int fof = n * this->shape[1] * fil_size;
        for (int ch = 0; ch < this->shape[1]; ch++) {
            for (int i = 0; i < nrows; i++) {
                for (int k = 0; k < fil_height; k++) {
                    int x_row = (i - padding[0]) * stride[0] + k;
                    if (x_row >= x_height)
                        break;
                    if (x_row < 0)
                        continue;
                    for (int j = 0; j < ncols; j++) {
                        for (int l = 0; l < fil_width; l++) {
                            int x_col = (j - padding[1]) * stride[1] + l;
                            if (x_col >= x_width)
                                break;
                            if (x_col >= 0) {
                                float x_val = this->data[ch * x_size + x_width * x_row + x_col];
                                float f_val = filter.data[fof + ch * fil_size + k * fil_height + l];

                                res_data[rof + i * nrows + j] += x_val * f_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return Tensor({1, filter.shape[0], nrows, ncols}, res_data);
}

Tensor Tensor::pad(DimVec padding, float value) {
    int hpad = padding[0];
    int wpad = padding[1];

    Tensor res_tensor =
        Tensor({this->shape[0], this->shape[1], this->shape[2] + hpad * 2, this->shape[3] + wpad * 2}, value);

    for (int n = 0; n < res_tensor.shape[0]; n++) {
        int ni = n * res_tensor.shape[1] * res_tensor.shape[2] * res_tensor.shape[3];
        int tni = n * this->shape[1] * this->shape[2] * this->shape[3];
        for (int c = 0; c < res_tensor.shape[1]; c++) {
            int ci = c * res_tensor.shape[2] * res_tensor.shape[3];
            int tci = c * this->shape[2] * this->shape[3];
            for (int h = 0; h < res_tensor.shape[2]; h++) {
                int hi = h * res_tensor.shape[3];
                int thi = (h - hpad) * this->shape[3];
                for (int w = 0; w < res_tensor.shape[3]; w++) {
                    if (h < hpad || h >= this->shape[2] + hpad || w < wpad || w >= this->shape[3] + wpad)
                        continue;

                    res_tensor.data[ni + ci + hi + w] = this->data[tni + tci + thi + w - wpad];
                }
            }
        }
    }

    return res_tensor;
}

Tensor Tensor::rot180() {
    int m = this->dshape[0];
    int n = this->dshape[1];

    std::vector<float> res_data(this->data);
    for (int bs = 0; bs < this->shape[0]; bs++) {
        for (int c = 0; c < this->shape[1]; c++) {
            int off = bs * this->shape[1] * m * n + c * m * n;
            for (int i = 0; i < m; i++) {
                for (int j = 0, k = n - 1; j < k; j++, k--) {
                    std::swap(res_data[off + i * n + j], res_data[off + i * n + k]);
                }
            }

            for (int j = 0; j < n; j++) {
                for (int i = 0, k = m - 1; i < k; i++, k--) {
                    std::swap(res_data[off + i * n + j], res_data[off + k * n + j]);
                }
            }
        }
    }

    return Tensor(this->shape, res_data);
}

void Tensor::backward() {
    if (this->backward_fn != nullptr) {
        this->backward_fn();
    }

    for (std::vector<Tensor*>::iterator it = this->children.begin(), end = this->children.end(); it != end; it++) {
        (*it)->backward();
    }
}

std::string Tensor::to_string() {
    std::string res = "tensor(";
    std::string data_str = "[";

    for (size_t i = 0; i < data.size(); i++) {
        data_str += std::to_string(data[i]);
        data_str += (i == data.size() - 1) ? "]" : ", ";

        if (i == 20) {
            data_str += "... ]";
            break;
        }
    }

    for (int shape : shape) {
        res += std::to_string(shape) + ",";
    }
    res.pop_back();

    return res + ") " + data_str;
}
