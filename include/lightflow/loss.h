#include "tensor.h"
#include <functional>

#ifndef LOSS_H
#define LOSS_H

Tensor cross_entropy_loss(Tensor* input, Tensor* target);
Tensor softmax_cross_entropy_loss(Tensor* input, Tensor* target);

Tensor l2_loss(Tensor* pred, Tensor* target);

#endif
