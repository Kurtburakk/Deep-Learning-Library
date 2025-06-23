#include "CaberNet/tensor.h"
#include "CaberNet/functions.h"
#include "internals/internal_tensor.hpp"
#include "internals/functions/internal_functions.hpp"

#include <iostream>
#include <memory>

namespace net::function {

Tensor<float> linear(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias) {
    return Tensor<float>(std::make_shared<internal::Linear>( input.internal(), weight.internal(), bias.internal() ));
}
Tensor<float> conv2d(const Tensor<float>& input, const Tensor<float>& weight, const Tensor<float>& bias,
                     int stride, int padding) {
    return Tensor<float>(std::make_shared<internal::Conv2d>( 
        input.internal(), weight.internal(), bias.internal(), stride, padding ));
}
Tensor<float> softmax(Tensor<float>& input, int axis) {
    return Tensor<float>(std::make_shared<internal::Softmax>( input.internal(), axis ));
}

Tensor<float> log_softmax(Tensor<float>& input, int axis) {
    return Tensor<float>(std::make_shared<internal::LogSoftmax>( input.internal(), axis ));
}

Tensor<float> relu(const Tensor<float>& input) {
    return Tensor<float>(std::make_shared<internal::ReLU>( input.internal() ));
}
Tensor<float> maxpool2d(const Tensor<float>& input, int kernel_size, int stride) {
    return Tensor<float>(std::make_shared<internal::MaxPool2d>( 
        input.internal(), kernel_size, stride ));
}
Tensor<float> flatten(const Tensor<float>& input, int start_dim) {
    return Tensor<float>(std::make_shared<internal::Flatten>( 
        input.internal(), start_dim ));
}

} // namespace net::function