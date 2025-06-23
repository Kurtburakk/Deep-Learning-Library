#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)

namespace internal {

Flatten::Flatten(Tensor* input, int start_dim)
:   Function(input)
,   start_dim_(start_dim < 0 ? input->rank() + start_dim : start_dim)
,   original_shape_(input->shape()) {
    
    // Validation
    if (start_dim_ < 0 || start_dim_ >= static_cast<int>(input->rank())) {
        throw std::runtime_error("Invalid start_dim for Flatten");
    }
    
    // Calculate output shape
    shape_type output_shape;
    
    // Keep dimensions before start_dim unchanged
    for (int i = 0; i < start_dim_; ++i) {
        output_shape.push_back(input->shape()[i]);
    }
    
    // Flatten dimensions from start_dim onwards
    size_type flattened_size = 1;
    for (size_type i = start_dim_; i < input->rank(); ++i) {
        flattened_size *= input->shape()[i];
    }
    output_shape.push_back(flattened_size);
    
    // Reshape output tensor
    reshape(output_shape);
    
    // Gradient requirement
    requires_gradient(input->requires_gradient());
}

Tensor* Flatten::forward() {
    const scalar_type* input_data = input()->forward()->data();
    scalar_type* output_data = this->data();
    
    // Simple memcpy since we're just reshaping
    size_type total_elements = input()->size();
    std::copy(input_data, input_data + total_elements, output_data);
    
    return this;
}

void Flatten::backward(Tensor* gradient) const {
    if (!input()->requires_gradient()) return;
    
    const scalar_type* gradient_data = gradient->data();
    
    // Create input gradient with original shape
    Tensor* input_gradient = new Tensor(original_shape_, false, false);
    scalar_type* input_grad_data = input_gradient->data();
    
    // Copy gradient data back (just reshaping)
    size_type total_elements = gradient->size();
    std::copy(gradient_data, gradient_data + total_elements, input_grad_data);
    
    input()->backward(input_gradient);
    delete input_gradient;
}

} // namespace internal

#endif // USE_EIGEN_BACKEND