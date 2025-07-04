/*
This classes are non-leaf nodes representing functions. They are used to build the computational.
A function should be implemented overriding the forward and backward methods, of the Tensor Base class.
The forward method should return the "this" pointer, and the backward method should pass the gradient
to the inputs.

Inplace functions are planned to be implemented in the future.

For implementing operations use the data() method to access the data of the tensor and map it to some
data structure of some library. In this case I used Eigen::Map so avoid any unnecessary copy of the data
when performing the operations, and to be able to use the optimized Eigen library for the operations.
*/


#ifndef INTERNAL_FUNCTIONS_HPP
#define INTERNAL_FUNCTIONS_HPP

#include "../internal_tensor.hpp"
#include "../internal_expression.hpp"

namespace internal {

class Function : public Expression {
    public:
    ~Function() override = default;
    Function(Tensor* input) { input_ = input; }
    Tensor* input() const { return input_; }

    private:
    Tensor* input_;
};

class Linear : public Function {
    public:
    ~Linear() final = default;
    Linear(Tensor* input, Tensor* weight, Tensor* bias);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    Tensor* weight() const { return weight_; }
    Tensor* bias() const { return bias_; }   

    size_type rows_dimension() const { return input()->shape().front(); }
    size_type inner_dimension() const { return input()->shape().back(); }
    size_type columns_dimension() const { return weight()->shape().front(); }

    private:
    std::unique_ptr<Tensor> weight_gradient_copy_;
    std::unique_ptr<Tensor> bias_gradient_copy_;
    Tensor* weight_;
    Tensor* bias_;       
};

class ReLU : public Function {
    public:
    ~ReLU() final = default;
    ReLU(Tensor* input);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;
};

class Softmax : public Function {
    public:
    ~Softmax() final = default;
    Softmax(Tensor* input, int axis);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    private:
    int axis_;
};

class LogSoftmax : public Function {
    public:
    ~LogSoftmax() final = default;
    LogSoftmax(Tensor* input, int axis);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    private:
    int axis_;
};

class Conv2d : public Function {
    public:
    ~Conv2d() final = default;
    Conv2d(Tensor* input, Tensor* weight, Tensor* bias, int stride = 1, int padding = 0);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    Tensor* weight() const { return weight_; }
    Tensor* bias() const { return bias_; }
    int stride() const { return stride_; }
    int padding() const { return padding_; }

    private:
    Tensor* weight_;
    Tensor* bias_;
    int stride_, padding_;
    
    // Cached dimensions
    size_type batch_size_, in_channels_, out_channels_;
    size_type input_height_, input_width_;
    size_type output_height_, output_width_;
    size_type kernel_height_, kernel_width_;
    
    // Gradient storage
    std::unique_ptr<Tensor> weight_gradient_copy_;
    std::unique_ptr<Tensor> bias_gradient_copy_;
    
    // Helper methods
    void im2col_cpu(const scalar_type* input_data, scalar_type* col_data) const;
    void col2im_cpu(const scalar_type* col_data, scalar_type* input_data) const;
};
class MaxPool2d : public Function {
    public:
    ~MaxPool2d() final = default;
    MaxPool2d(Tensor* input, int kernel_size, int stride = -1);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    int kernel_size() const { return kernel_size_; }
    int stride() const { return stride_; }

    private:
    int kernel_size_, stride_;
    
    // Cached dimensions
    size_type batch_size_, channels_;
    size_type input_height_, input_width_;
    size_type output_height_, output_width_;
    
    // For backward pass - store max indices
    std::unique_ptr<std::vector<size_type>> max_indices_;
};
class Flatten : public Function {
    public:
    ~Flatten() final = default;
    Flatten(Tensor* input, int start_dim = 1);
    Tensor* forward() final;
    void backward(Tensor* gradient) const final;

    int start_dim() const { return start_dim_; }

    private:
    int start_dim_;
    shape_type original_shape_;  // Store original shape for backward pass
};

} // namespace internal

#endif // INTERNAL_FUNCTIONS_HPP