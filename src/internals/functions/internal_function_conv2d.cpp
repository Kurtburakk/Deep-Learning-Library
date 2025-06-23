#include "../config.h"
#include "../internal_array.hpp"
#include "../internal_tensor.hpp"
#include "internal_functions.hpp"

#if defined(USE_EIGEN_BACKEND)


namespace internal {

Conv2d::Conv2d(Tensor* input, Tensor* weight, Tensor* bias, int stride, int padding)
:   Function(input)
,   weight_(weight)
,   bias_(bias)
,   stride_(stride)
,   padding_(padding) {
    
    // Input validation (matching your Linear style)
    if (input->rank() != 4) throw std::runtime_error("Input must be 4D: [N,C,H,W]");
    if (weight->rank() != 4) throw std::runtime_error("Weight must be 4D: [out_ch,in_ch,kH,kW]");
    if (bias && bias->rank() != 2) throw std::runtime_error("Bias must be 2D: [1,out_ch]");
    
    // Shape calculation
    batch_size_ = input->shape()[0];
    in_channels_ = input->shape()[1];
    input_height_ = input->shape()[2];
    input_width_ = input->shape()[3];
    
    out_channels_ = weight->shape()[0];
    if (weight->shape()[1] != in_channels_) throw std::runtime_error("shape mismatch between input and weight channels");
    kernel_height_ = weight->shape()[2];
    kernel_width_ = weight->shape()[3];
    
    if (bias && bias->shape()[1] != out_channels_) throw std::runtime_error("shape mismatch between bias and weight");
    
    // Output dimensions calculation
    output_height_ = (input_height_ + 2*padding - kernel_height_) / stride + 1;
    output_width_ = (input_width_ + 2*padding - kernel_width_) / stride + 1;
    
    // Reshape output tensor
    reshape({batch_size_, out_channels_, output_height_, output_width_});
    
    // Gradient requirement (your exact pattern from Linear)
    bool gradient_requirement = input->requires_gradient() || 
                               weight->requires_gradient() || 
                               (bias && bias->requires_gradient());
    requires_gradient(gradient_requirement);
    
    // Gradient storage (your exact pattern from Linear)
    if (weight->requires_gradient()) {
        weight_gradient_copy_ = std::make_unique<Tensor>(weight->shape(), false, false);
    }
    if (bias && bias->requires_gradient()) {
        bias_gradient_copy_ = std::make_unique<Tensor>(bias->shape(), false, false);
    }
}

Tensor* Conv2d::forward() {
    // Get input data
    const scalar_type* input_data = input()->forward()->data();
    const scalar_type* weight_data = weight()->forward()->data();
    scalar_type* output_data = this->data();
    
    // im2col dimensions
    size_type col_height = in_channels_ * kernel_height_ * kernel_width_;
    size_type col_width = output_height_ * output_width_;
    
    for (size_type batch = 0; batch < batch_size_; ++batch) {
        // Temporary column matrix for this batch item
        std::vector<scalar_type> col_buffer(col_height * col_width);
        
        // Convert input patch to column matrix (im2col)
        const scalar_type* batch_input = input_data + batch * (in_channels_ * input_height_ * input_width_);
        im2col_cpu(batch_input, col_buffer.data());
        
        // Map to Eigen matrices (your exact style from Linear)
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> weight_matrix(
            weight_data,
            out_channels_,
            col_height
        );
        
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> col_matrix(
            col_buffer.data(),
            col_height,
            col_width
        );
        
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> output_matrix(
            output_data + batch * (out_channels_ * output_height_ * output_width_),
            out_channels_,
            col_width
        );
        
        // Matrix multiplication: output = weight @ col_matrix (your Linear style)
        output_matrix = weight_matrix * col_matrix;
        
        // Add bias if present (broadcasting, matching your Linear implementation)
        if (bias()) {
            Eigen::Map<const Eigen::Matrix<scalar_type, 1, -1>> bias_vector(
                bias()->forward()->data(), 1, out_channels_
            );
            
            output_matrix.colwise() += bias_vector.transpose();
        }
    }
    
    return this;
}

void Conv2d::backward(Tensor* gradient) const {
    const scalar_type* gradient_data = gradient->data();
    const scalar_type* input_data = input()->data();
    const scalar_type* weight_data = weight()->data();
    
    // Gradient w.r.t input (following your Linear backward pattern)
    if (input()->requires_gradient()) {
        Tensor* input_gradient = new Tensor(input()->shape(), false, false);
        scalar_type* input_grad_data = input_gradient->data();
        
        // Zero initialize
        std::fill(input_grad_data, input_grad_data + input_gradient->size(), 0.0f);
        
        for (size_type batch = 0; batch < batch_size_; ++batch) {
            // Get gradient for this batch
            const scalar_type* batch_gradient = gradient_data + batch * (out_channels_ * output_height_ * output_width_);
            scalar_type* batch_input_grad = input_grad_data + batch * (in_channels_ * input_height_ * input_width_);
            
            // Map gradient as matrix (your Eigen style)
            Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> grad_matrix(
                batch_gradient, out_channels_, output_height_ * output_width_
            );
            
            Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> weight_matrix(
                weight_data, out_channels_, in_channels_ * kernel_height_ * kernel_width_
            );
            
            // Compute gradient w.r.t column matrix
            Eigen::Matrix<scalar_type, -1, -1, 1> col_grad = weight_matrix.transpose() * grad_matrix;
            
            // Convert back to input space using col2im
            col2im_cpu(col_grad.data(), batch_input_grad);
        }
        
        input()->backward(input_gradient);
        delete input_gradient;
    }
    
    // Gradient w.r.t weight (your exact pattern from Linear)
    if (weight()->requires_gradient()) {
        Eigen::Map<Eigen::Matrix<scalar_type, -1, -1, 1>> weight_grad_matrix(
            weight_gradient_copy_->data(),
            out_channels_,
            in_channels_ * kernel_height_ * kernel_width_
        );
        
        weight_grad_matrix.setZero();
        
        for (size_type batch = 0; batch < batch_size_; ++batch) {
            // Get input and gradient for this batch
            const scalar_type* batch_input = input_data + batch * (in_channels_ * input_height_ * input_width_);
            const scalar_type* batch_gradient = gradient_data + batch * (out_channels_ * output_height_ * output_width_);
            
            // Convert input to column matrix
            std::vector<scalar_type> col_buffer(in_channels_ * kernel_height_ * kernel_width_ * output_height_ * output_width_);
            im2col_cpu(batch_input, col_buffer.data());
            
            Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> col_matrix(
                col_buffer.data(),
                in_channels_ * kernel_height_ * kernel_width_,
                output_height_ * output_width_
            );
            
            Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> grad_matrix(
                batch_gradient, out_channels_, output_height_ * output_width_
            );
            
            // Accumulate gradient
            weight_grad_matrix += grad_matrix * col_matrix.transpose();
        }
        
        weight()->backward(weight_gradient_copy_.get());
    }
    
    // Gradient w.r.t bias (your exact pattern from Linear)
    if (bias() && bias()->requires_gradient()) {
        Eigen::Map<Eigen::Matrix<scalar_type, 1, -1>> bias_gradient_map(
            bias_gradient_copy_->data(), 1, out_channels_
        );
        
        Eigen::Map<const Eigen::Matrix<scalar_type, -1, -1, 1>> gradient_map(
            gradient_data,
            out_channels_,
            batch_size_ * output_height_ * output_width_
        );
        
        bias_gradient_map = gradient_map.rowwise().sum();
        bias()->backward(bias_gradient_copy_.get());
    }
}

// Helper method: im2col (input to column matrix)
void Conv2d::im2col_cpu(const scalar_type* input_data, scalar_type* col_data) const {
    for (size_type c = 0; c < in_channels_; ++c) {
        for (size_type kh = 0; kh < kernel_height_; ++kh) {
            for (size_type kw = 0; kw < kernel_width_; ++kw) {
                size_type col_row = c * kernel_height_ * kernel_width_ + kh * kernel_width_ + kw;
                
                for (size_type oh = 0; oh < output_height_; ++oh) {
                    for (size_type ow = 0; ow < output_width_; ++ow) {
                        size_type col_col = oh * output_width_ + ow;
                        
                        // Calculate input position
                        int ih = oh * stride_ - padding_ + kh;
                        int iw = ow * stride_ - padding_ + kw;
                        
                        // Check bounds
                        if (ih >= 0 && ih < static_cast<int>(input_height_) && 
                            iw >= 0 && iw < static_cast<int>(input_width_)) {
                            size_type input_idx = c * input_height_ * input_width_ + ih * input_width_ + iw;
                            col_data[col_row * output_height_ * output_width_ + col_col] = input_data[input_idx];
                        } else {
                            col_data[col_row * output_height_ * output_width_ + col_col] = 0.0f; // Padding
                        }
                    }
                }
            }
        }
    }
}

// Helper method: col2im (column matrix to input)
void Conv2d::col2im_cpu(const scalar_type* col_data, scalar_type* input_data) const {
    for (size_type c = 0; c < in_channels_; ++c) {
        for (size_type kh = 0; kh < kernel_height_; ++kh) {
            for (size_type kw = 0; kw < kernel_width_; ++kw) {
                size_type col_row = c * kernel_height_ * kernel_width_ + kh * kernel_width_ + kw;
                
                for (size_type oh = 0; oh < output_height_; ++oh) {
                    for (size_type ow = 0; ow < output_width_; ++ow) {
                        size_type col_col = oh * output_width_ + ow;
                        
                        // Calculate input position
                        int ih = oh * stride_ - padding_ + kh;
                        int iw = ow * stride_ - padding_ + kw;
                        
                        // Check bounds and accumulate
                        if (ih >= 0 && ih < static_cast<int>(input_height_) && 
                            iw >= 0 && iw < static_cast<int>(input_width_)) {
                            size_type input_idx = c * input_height_ * input_width_ + ih * input_width_ + iw;
                            input_data[input_idx] += col_data[col_row * output_height_ * output_width_ + col_col];
                        }
                    }
                }
            }
        }
    }
}

} // namespace internal

#endif // USE_EIGEN_BACKEND