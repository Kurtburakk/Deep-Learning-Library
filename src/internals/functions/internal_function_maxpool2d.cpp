#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

MaxPool2d::MaxPool2d(Tensor* input, int kernel_size, int stride)
:   Function(input)
,   kernel_size_(kernel_size)
,   stride_(stride == -1 ? kernel_size : stride) {  // Default stride = kernel_size
    
    // Input validation (following your Conv2d style)
    if (input->rank() != 4) throw std::runtime_error("Input must be 4D: [N,C,H,W]");
    
    // Shape calculation
    batch_size_ = input->shape()[0];
    channels_ = input->shape()[1];
    input_height_ = input->shape()[2];
    input_width_ = input->shape()[3];
    
    // Output dimensions calculation
    output_height_ = (input_height_ - kernel_size_) / stride_ + 1;
    output_width_ = (input_width_ - kernel_size_) / stride_ + 1;
    
    // Validate output dimensions
    if (output_height_ <= 0 || output_width_ <= 0) {
        throw std::runtime_error("Invalid pooling parameters: output size would be non-positive");
    }
    
    // Reshape output tensor
    reshape({batch_size_, channels_, output_height_, output_width_});
    
    // Gradient requirement (MaxPool2d passes through gradients)
    requires_gradient(input->requires_gradient());
    
    // Initialize max indices storage for backward pass
    if (input->requires_gradient()) {
        size_type total_output_elements = batch_size_ * channels_ * output_height_ * output_width_;
        max_indices_ = std::make_unique<std::vector<size_type>>(total_output_elements);
    }
}

Tensor* MaxPool2d::forward() {
    const scalar_type* input_data = input()->forward()->data();
    scalar_type* output_data = this->data();
    
    // Process each batch and channel independently
    for (size_type batch = 0; batch < batch_size_; ++batch) {
        for (size_type ch = 0; ch < channels_; ++ch) {
            
            // Calculate base indices for this batch and channel
            size_type input_base = batch * (channels_ * input_height_ * input_width_) + 
                                  ch * (input_height_ * input_width_);
            size_type output_base = batch * (channels_ * output_height_ * output_width_) + 
                                   ch * (output_height_ * output_width_);
            
            // Apply max pooling
            for (size_type oh = 0; oh < output_height_; ++oh) {
                for (size_type ow = 0; ow < output_width_; ++ow) {
                    
                    // Find max value in the pooling window
                    scalar_type max_val = -std::numeric_limits<scalar_type>::infinity();
                    size_type max_idx = 0;
                    
                    for (size_type kh = 0; kh < static_cast<size_type>(kernel_size_); ++kh) {
                        for (size_type kw = 0; kw < static_cast<size_type>(kernel_size_); ++kw) {
                            size_type ih = oh * stride_ + kh;
                            size_type iw = ow * stride_ + kw;
                            
                            // Check bounds (should be valid due to our dimension calculation)
                            if (ih < input_height_ && iw < input_width_) {
                                size_type input_idx = input_base + ih * input_width_ + iw;
                                scalar_type val = input_data[input_idx];
                                
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }
                    
                    // Store max value
                    size_type output_idx = output_base + oh * output_width_ + ow;
                    output_data[output_idx] = max_val;
                    
                    // Store max index for backward pass
                    if (max_indices_) {
                        size_type global_output_idx = batch * (channels_ * output_height_ * output_width_) + 
                                                     ch * (output_height_ * output_width_) + 
                                                     oh * output_width_ + ow;
                        (*max_indices_)[global_output_idx] = max_idx;
                    }
                }
            }
        }
    }
    
    return this;
}

void MaxPool2d::backward(Tensor* gradient) const {
    if (!input()->requires_gradient()) return;
    
    const scalar_type* gradient_data = gradient->data();
    
    // Create input gradient tensor (zero initialized)
    Tensor* input_gradient = new Tensor(input()->shape(), false, false);
    scalar_type* input_grad_data = input_gradient->data();
    std::fill(input_grad_data, input_grad_data + input_gradient->size(), 0.0f);
    
    // Distribute gradients back to max locations
    size_type total_output_elements = batch_size_ * channels_ * output_height_ * output_width_;
    
    for (size_type i = 0; i < total_output_elements; ++i) {
        size_type max_input_idx = (*max_indices_)[i];
        input_grad_data[max_input_idx] += gradient_data[i];
    }
    
    input()->backward(input_gradient);
    delete input_gradient;
}

} // namespace internal

#endif // USE_EIGEN_BACKEND