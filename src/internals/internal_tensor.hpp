
#ifndef INTERNAL_TENSOR_HPP
#define INTERNAL_TENSOR_HPP

#include <iostream>
#include <vector>
#include <memory>

#include "internal_array.hpp"

namespace internal {

class Tensor : public Array<float> {
    public:
    Tensor(bool leaf_status = true)
    :   Array() {
        is_leaf_ = leaf_status; 
        requires_gradient_ = false;
    }

    Tensor(shape_type shape, bool gradient_requirement = false, bool leaf_status = true) 
    :   Array(shape) {
        is_leaf_ = leaf_status; 
        requires_gradient_ = gradient_requirement;
        if (is_leaf_ && requires_gradient_) gradient_ = new Tensor(shape, false, false);
    }

    virtual ~Tensor() { if (is_leaf_ && requires_gradient_) delete gradient_; }
    Tensor(const Tensor* other) { copy(other); }
    Tensor(const Tensor& other) = delete;
    Tensor(Tensor&& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
    Tensor& operator=(Tensor&& other) = delete;


    void copy(const Tensor* other) {
        reshape(other->shape());
        std::copy(other->begin(), other->end(), this->begin());
        requires_gradient_ = other->requires_gradient_;

        if (requires_gradient_ ) {
            if (other->is_leaf_ && is_leaf_) {
                if (!gradient_) gradient_ = new Tensor(other->gradient_);
                else gradient_->copy(other->gradient_);
            }
            
            else {
                if (is_leaf_) delete gradient_;
                gradient_ = other->gradient_;
            }

        }
        
        else {
            if (is_leaf_) delete gradient_;
            gradient_ = nullptr;
        }

        is_leaf_ = other->is_leaf_;
    }

    void move(Tensor* other) {
        reshape(other->shape());
        std::move(other->begin(), other->end(), this->begin());
        other->clear();
        if (is_leaf_) delete gradient_;
        is_leaf_ = other->is_leaf_;
        requires_gradient_ = other->requires_gradient_;
        gradient_ = other->gradient_;
        other->gradient_ = nullptr;
    }

    Tensor* gradient() const { return gradient_; }

    void requires_gradient(bool status) {        
        if (requires_gradient_ == false && status == true) {
            requires_gradient_ = true;
            if (is_leaf_) gradient_ = new Tensor(this->shape(), false, false);
        }

        if (requires_gradient_ == true && status == false ) {
            requires_gradient_ = false;
            if (is_leaf_) delete gradient_;
            gradient_ = nullptr;
        }
    }

    bool is_leaf() const { return is_leaf_; }
    bool requires_gradient() const { return requires_gradient_; }

    virtual Tensor* forward() { return this; }
    virtual void backward(Tensor* gradient) const { gradient_->add(gradient); }

    void add(const Tensor* other);
    void multiply(const Tensor* other);

    protected:
    bool is_leaf_;
    bool requires_gradient_;

    private:
    Tensor* gradient_;
};

} // namespace internal

#endif // INTERNAL_TENSOR_HPP