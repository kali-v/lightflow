#include "loss.h"

#include <math.h>

#include <functional>
#include <iostream>
#include <ostream>

#include "diff.h"
#include "tensor.h"

void check_same_dims(Tensor* pred, Tensor* target) {
    if (pred->shape != target->shape) {
        throw std::runtime_error("l2_loss: different diffs");
    }
}

Tensor softmax_cross_entropy_loss(Tensor* pred, Tensor* target) {
    Tensor z_tensor = *pred - pred->max();
    Tensor exp_tensor = z_tensor.apply((float (*)(float))std::exp);
    Tensor pred_tensor = exp_tensor / exp_tensor.sum();
    pred_tensor += Tensor(pred_tensor.shape, 1e-8f);
    pred->data = pred_tensor.data;

    Tensor log_tensor = pred_tensor.apply((float (*)(float))std::log);
    Tensor mul_tensor = *target * log_tensor;

    Tensor out = Tensor({1, 1}, mul_tensor.sum() * -1, {pred, target}, true);
    out.backward_fn = softmax_cross_entropy_backward(pred, target);

    return out;
}

Tensor cross_entropy_loss(Tensor* pred, Tensor* target, bool used_softmax) {
    Tensor log_tensor = pred->apply((float (*)(float))std::log);
    Tensor mul_tensor = *target * *pred;

    Tensor out = Tensor({1, 1}, mul_tensor.sum() * -1, {pred, target}, true);
    out.backward_fn = cross_entropy_backward(pred, target);

    return out;
}

Tensor l2_loss(Tensor* pred, Tensor* target) {
    check_same_dims(pred, target);

    Tensor diff_tensor = *pred - *target;
    Tensor pow_tensor = diff_tensor.pow(2);

    Tensor out = Tensor({1, 1}, pow_tensor.sum(), {pred, target}, true);
    out.backward_fn = l2_backward(pred, target);

    return out;
}
