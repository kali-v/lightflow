#include <cmath>
#include <functional>
#include <iostream>
#include <ostream>

#include "tensor.h"

std::function<void()> add_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        a->add_grad(out->grad->data);
        b->add_grad(out->grad->data);
    };
}

std::function<void()> sub_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        a->add_grad(out->grad->data);
        b->add_grad((*out->grad * -1).data);
    };
}

std::function<void()> mul_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        a->add_grad((*b * *out->grad).data);
        b->add_grad((*a * *out->grad).data);
    };
}

std::function<void()> ddiv_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        Tensor a_grad_tensor = b->pow(-1);

        Tensor pow_tensor = b->pow(2);
        Tensor b_grad_tensor = *a / pow_tensor * -1;

        a->add_grad((a_grad_tensor * *out->grad).data);
        b->add_grad((b_grad_tensor * *out->grad).data);
    };
}

std::function<void()> pow_backward(Tensor* a, Tensor* exp, Tensor* out) {
    return [a, exp, out]() {
        Tensor a_grad_tensor = (*exp * *a).pow(exp->data[0] - 1);

        Tensor log_tensor = a->apply((float (*)(float))std::log);
        Tensor pow_tensor = a->pow(exp->data[0]);

        a->add_grad((a_grad_tensor * *out->grad).data);
        exp->add_grad((log_tensor * pow_tensor * *out->grad).data);
    };
}

std::function<void()> matmul_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        Tensor a_trans = a->transpose();
        Tensor b_trans = b->transpose();

        a->add_grad((out->grad->matmul(b_trans)).data);
        b->add_grad((a_trans.matmul(*out->grad)).data);
    };
}

std::function<void()> channelwise_sum_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() {
        a->add_grad(out->grad->data);

        Vec1D grad(b->data.size(), 0.0f);
        int out_wh = out->shape[2] * out->shape[3];
        for (int c = 0; c < out->shape[1]; c++) {
            for (int i = 0; i < out_wh; i++) {
                grad[c] += out->grad->data[c * out_wh + i];
            }
        }
        b->add_grad(grad);
    };
}

std::function<void()> pad_backward(Tensor* x, DimVec padding, Tensor* out) {
    return [x, padding, out]() {
        int hpad = padding[0];
        int wpad = padding[1];

        Vec1D grad;
        grad.reserve(x->data.size());

        for (int n = 0; n < out->shape[0]; n++) {
            int ni = n * out->shape[1] * out->shape[2] * out->shape[3];
            for (int c = 0; c < out->shape[1]; c++) {
                int ci = c * out->shape[2] * out->shape[3];
                for (int h = 0; h < out->shape[2]; h++) {
                    if (h < hpad)
                        continue;
                    if (h >= hpad + x->shape[2])
                        break;
                    int hi = h * out->shape[3];
                    for (int w = 0; w < out->shape[3]; w++) {
                        if (w >= wpad && w < wpad + x->shape[3]) {
                            grad.push_back(out->grad->data[ni + ci + hi + w]);
                        }
                    }
                }
            }
        }
        x->add_grad(grad);
    };
}

std::function<void()> correlate_backward(Tensor* x, Tensor* filter, Tensor* out, DimVec stride) {
    return [x, filter, out, stride]() {
        int dil = stride[0];
        int dil_p = dil > 1 ? dil - 1 : 0;
        int rest = (x->shape[3] - filter->shape[3]) % stride[0] - dil_p;

        // pad out gradient
        Tensor pad_out =
            Tensor({out->shape[0], out->shape[1], out->shape[2] * dil + rest, out->shape[3] * dil + rest}, 0.0f);

        int pad_out_size = pad_out.dshape[0] * pad_out.dshape[1];
        int out_size = out->dshape[0] * out->dshape[1];

        for (int n = 0; n < pad_out.shape[0]; n++) {
            int nmi = n * pad_out.shape[1] * pad_out_size;
            int ni = n * out->shape[1] * out_size;
            for (int c = 0; c < pad_out.shape[1]; c++) {
                int cmi = c * pad_out_size;
                int ci = c * out_size;
                for (int j = 0; j < pad_out.shape[2]; j += dil) {
                    int jmi = j * pad_out.dshape[1];
                    int ji = (j / dil) * out->dshape[1];
                    for (int i = 0; i < pad_out.shape[3]; i += dil) {
                        pad_out.data[nmi + cmi + jmi + i] = out->grad->data[ni + ci + ji + i / dil];
                    }
                }
            }
        }

        int fsize = filter->dshape[0] * filter->dshape[1];

        if (x->require_grad) {
            Tensor* rot_filter = new Tensor(*filter);
            *rot_filter = rot_filter->rot180();

            Vec1D x_grad(x->data.size(), 0.0f);
            Vec1D fil_grad(filter->data.size(), 0.0f);

            int p = filter->shape[3] - 1;
            int x_size = x->dshape[0] * x->dshape[1];

            // correlate rotated filter over padded output gradient
            for (int b = 0; b < pad_out.shape[0]; b++) {
                Tensor pad_block = pad_out.get_block(b);
                Tensor xblock = x->get_block(b);
                int offb = b * x->shape[1] * x_size;

                for (int ch = 0; ch < rot_filter->shape[0]; ch++) {
                    Tensor pad_chan = pad_block.get_channel(ch);
                    Tensor rot_block = rot_filter->get_block(ch);
                    for (int fc = 0; fc < rot_filter->shape[1]; fc++) {
                        Tensor fil_chan = rot_block.get_channel(fc);
                        Vec1D cor_data = pad_chan.correlate(fil_chan, {1, 1}, {p, p}).data;

                        auto xgrad_beg = x_grad.begin() + fc * x_size + offb;
                        std::transform(xgrad_beg, xgrad_beg + cor_data.size(), cor_data.begin(), xgrad_beg,
                                       std::plus<float>());
                    }
                    for (int xch = 0; xch < x->shape[1]; xch++) {
                        Vec1D cor_data = xblock.get_channel(xch).correlate(pad_chan).data;

                        auto filgrad_beg = fil_grad.begin() + ch * filter->shape[1] * fsize + xch * fsize;
                        std::transform(filgrad_beg, filgrad_beg + cor_data.size(), cor_data.begin(), filgrad_beg,
                                       std::plus<float>());
                    }
                }
            }
            x->add_grad(x_grad);
            filter->add_grad(fil_grad);
        } else {
            // compute filter grad
            Vec1D fil_grad(filter->data.size(), 0.0f);

            for (int b = 0; b < pad_out.shape[0]; b++) {
                Tensor pad_block = pad_out.get_block(b);
                Tensor xblock = x->get_block(b);
                for (int ch = 0; ch < pad_out.shape[1]; ch++) {
                    Tensor pad_chan = pad_block.get_channel(ch);
                    for (int xch = 0; xch < x->shape[1]; xch++) {
                        Vec1D cor_data = xblock.get_channel(xch).correlate(pad_chan).data;

                        auto filgrad_beg = fil_grad.begin() + ch * filter->shape[1] * fsize + xch * fsize;
                        std::transform(filgrad_beg, filgrad_beg + cor_data.size(), cor_data.begin(), filgrad_beg,
                                       std::plus<float>());
                    }
                }
            }

            filter->add_grad(fil_grad);
        }
    };
}

std::function<void()> reshape_backward(Tensor* x, Tensor* out) {
    return [x, out]() { x->add_grad(out->grad->reshape(x->shape).data); };
}

std::function<void()> maxpool2d_backward(Tensor* x, Tensor* out, std::vector<int> argmaxs) {
    return [x, out, argmaxs]() {
        Vec1D grad(x->data.size(), 0.0f);
        for (std::size_t i = 0; i < argmaxs.size(); i++) {
            grad[argmaxs[i]] = out->grad->data[i];
        }
        x->add_grad(grad);
    };
}

std::function<void()> l2_backward(Tensor* pred, Tensor* target) {
    return [pred, target]() {
        Tensor grad_tensor = (*pred - *target) * 2;
        pred->add_grad(grad_tensor.data);
        target->add_grad(grad_tensor.data);
    };
}

std::function<void()> softmax_cross_entropy_backward(Tensor* pred, Tensor* target) {
    return [pred, target]() {
        Tensor grad_tensor = *pred - *target;
        pred->add_grad(grad_tensor.data);
        target->add_grad(grad_tensor.data);
    };
}

std::function<void()> cross_entropy_backward(Tensor* pred, Tensor* target) {
    return [pred, target]() {
        Tensor grad_tensor = (*pred - *target) * 2;
        pred->add_grad(grad_tensor.data);
        target->add_grad(grad_tensor.data);
    };
}

std::function<void()> relu_backward(Tensor* x, Tensor* out) {
    return [x, out]() {
        std::vector<float> data;
        for (std::size_t i = 0; i < x->data.size(); i++) {
            data.push_back((x->data[i] > 0) * out->grad->data[i]);
        }

        x->add_grad(data);
    };
}

std::function<void()> leaky_relu_backward(Tensor* x, Tensor* out, float negative_slope) {
    return [x, out, negative_slope]() {
        std::vector<float> data(x->data.size());
        for (std::size_t i = 0; i < x->data.size(); i++) {
            float slope = (x->data[i] > 0) ? 1 : negative_slope;
            data[i] = slope * out->grad->data[i];
        }

        x->add_grad(data);
    };
}

std::function<void()> sigmoid_backward(Tensor* x, Tensor* out) {
    return [x, out]() {
        Tensor nx = *x * -1;
        Tensor enx = Tensor(nx.dshape, std::exp(1.0f)).pow(nx);
        Tensor den = (enx + 1).pow(2);
        Tensor grad = enx / den * *out->grad;
        x->add_grad(grad.data);
    };
}
