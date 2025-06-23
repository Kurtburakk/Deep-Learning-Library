#pragma once
#include "tensor.h"

namespace net::function {

Tensor<float> linear(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias);
Tensor<float> softmax(Tensor<float>& input, int axis);
Tensor<float> log_softmax(Tensor<float>&input, int axis);
Tensor<float> relu(const Tensor<float>& input);
Tensor<float> conv2d(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias, 
                     int stride = 1, int padding = 0);
Tensor<float> maxpool2d(const Tensor<float>& input, int kernel_size, int stride = -1);
Tensor<float> flatten(const Tensor<float>& input, int start_dim = 1);

} // namespace net::function
