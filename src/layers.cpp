#include "CaberNet/tensor.h"
#include "CaberNet/layers.h"

#include "internals/functions/internal_functions.hpp"


namespace net::layer {

/// constructors

Linear::~Linear() = default;

Linear::Linear(size_type input_features, size_type output_features, initializer distribution)
:   weight_(shape_type{output_features, input_features}, true),
    bias_(shape_type{1, output_features}, true) {
    weight_.fill(distribution);
    bias_.fill(0.0);
}

// 🆕 ADD Conv2d Implementation
Conv2d::~Conv2d() = default;

Conv2d::Conv2d(size_type in_channels, size_type out_channels, size_type kernel_size,
               size_type stride, size_type padding, initializer distribution)
:   weight_(shape_type{out_channels, in_channels, kernel_size, kernel_size}, true),
    bias_(shape_type{1, out_channels}, true),
    stride_(stride), padding_(padding), kernel_size_(kernel_size) {
    
    // He initialization for Conv2d (good for ReLU)
    weight_.fill(distribution);
    bias_.fill(0.0);  // Zero bias initialization
}

Softmax::Softmax(int axis) : axis(axis) {}
LogSoftmax::LogSoftmax(int axis) : axis(axis) {}

/// settings

void Linear::set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
    optimizer->add_parameter(weight_.internal());
    optimizer->add_parameter(bias_.internal());
}

// 🆕 ADD Conv2d optimizer setup
void Conv2d::set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
    optimizer->add_parameter(weight_.internal());
    optimizer->add_parameter(bias_.internal());
}

/// forward methods

Tensor<float> Linear::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Linear>(input.internal(), weight_.internal(), bias_.internal()));
}

// 🆕 ADD Conv2d forward method
Tensor<float> Conv2d::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Conv2d>(
        input.internal(), weight_.internal(), bias_.internal(), stride_, padding_));
}

// 🆕 MaxPool2d Implementation
MaxPool2d::~MaxPool2d() = default;

MaxPool2d::MaxPool2d(size_type kernel_size, size_type stride)
:   kernel_size_(kernel_size), 
    stride_(stride == 0 ? kernel_size : stride) {
}

Flatten::~Flatten() = default;

Flatten::Flatten(int start_dim) : start_dim_(start_dim) {}

void Flatten::set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
    return;  // No parameters
}

Tensor<float> Flatten::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Flatten>(
        input.internal(), start_dim_));
}

void MaxPool2d::set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
    return;  // No parameters
}

Tensor<float> MaxPool2d::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::MaxPool2d>(
        input.internal(), kernel_size_, stride_));
}

Tensor<float> ReLU::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::ReLU>(input.internal()));
}

Tensor<float> Softmax::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::Softmax>(input.internal(), axis));
}

Tensor<float> LogSoftmax::forward(Tensor<float> input) {
    return Tensor<float>(std::make_shared<internal::LogSoftmax>(input.internal(), axis));
}

} // namespace net::layer