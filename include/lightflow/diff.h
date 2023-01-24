#ifndef DIFF_H
#define DIFF_H

std::function<void()> add_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> sub_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> mul_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> ddiv_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> pow_backward(Tensor* a, Tensor* exp, Tensor* out);

std::function<void()> pad_backward(Tensor* x, DimVec padding, Tensor* out);

std::function<void()> matmul_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> channelwise_sum_backward(Tensor* a, Tensor* b, Tensor* out);
std::function<void()> correlate_backward(Tensor* x, Tensor* filter, Tensor* out, DimVec stride);
std::function<void()> reshape_backward(Tensor* x, Tensor* out);

std::function<void()> maxpool2d_backward(Tensor* x, Tensor* out, std::vector<int> argmaxs);

std::function<void()> l2_backward(Tensor* pred, Tensor* target);
std::function<void()> cross_entropy_backward(Tensor* pred, Tensor* target);
std::function<void()> softmax_cross_entropy_backward(Tensor* pred, Tensor* target);

std::function<void()> leaky_relu_backward(Tensor* x, Tensor* out, float negative_slope);
std::function<void()> relu_backward(Tensor* x, Tensor* out);

std::function<void()> sigmoid_backward(Tensor* x, Tensor* out);

#endif
