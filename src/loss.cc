#include "loss.h"

#include <math.h>

#include <functional>
#include <iostream>
#include <ostream>

#include "diff.h"
#include "tensor.h"

void check_same_dims(Tensor* pred, Tensor* target) {
    if (pred->shape_ != target->shape_) {
        throw std::runtime_error("l2_loss: different diffs");
    }
}

Tensor softmax_cross_entropy_loss(Tensor* pred, Tensor* target) {
    check_cpu(__func__, pred->device_);

    Tensor z_tensor = *pred - pred->max();
    Tensor exp_tensor = z_tensor.apply((float (*)(float))std::exp);
    Tensor pred_tensor = exp_tensor / exp_tensor.sum();
    pred_tensor += Tensor(pred_tensor.shape_, 1e-8f);
    pred->data_ = pred_tensor.data_;

    Tensor log_tensor = pred_tensor.apply((float (*)(float))std::log);
    Tensor mul_tensor = *target * log_tensor;

    Tensor out = Tensor({1, 1}, mul_tensor.sum() * -1, {pred, target}, true);
    out.backward_fn_ = softmax_cross_entropy_backward(pred, target);

    return out;
}

Tensor cross_entropy_loss(Tensor* pred, Tensor* target) {
    Tensor log_tensor = pred->log();
    Tensor mul_tensor = *target * *pred;

    Tensor out = Tensor({1, 1}, mul_tensor.sum() * -1, {pred, target}, true);
    out.backward_fn_ = cross_entropy_backward(pred, target);

    return out;
}

Tensor l2_loss(Tensor* pred, Tensor* target) {
    check_same_dims(pred, target);

    Tensor diff_tensor = *pred - *target;
    Tensor pow_tensor = diff_tensor.pow(2);

    Tensor out = Tensor({1, 1}, pow_tensor.sum(), {pred, target}, true);
    out.backward_fn_ = l2_backward(pred, target);

    return out;
}
