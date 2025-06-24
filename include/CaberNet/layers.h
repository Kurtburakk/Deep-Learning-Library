#pragma once

#include "model.h"

#include <iostream>
#include <memory>
#include <vector>
#include <variant>

namespace internal {
    class Tensor;
};

namespace net::layer {

class Linear : public Model<Linear> {
    public:
    ~Linear();
    Linear(
        size_type input_features,
        size_type output_features,
        initializer distribution = initializer::He );

    Tensor<float> forward(Tensor<float> x);

    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer);

    std::vector<internal::Tensor*> parameters() const {
        return { weight_.internal(), bias_.internal() };
    }
  
    private:
    Tensor<float> weight_;
    Tensor<float> bias_;
};

// 🆕 ADD Conv2d Layer Class
class Conv2d : public Model<Conv2d> {
    public:
    ~Conv2d();
    Conv2d(size_type in_channels, size_type out_channels, size_type kernel_size,
           size_type stride = 1, size_type padding = 0, 
           initializer distribution = initializer::He);

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer);

    std::vector<internal::Tensor*> parameters() const {
        return { weight_.internal(), bias_.internal() };
    }

    private:
    Tensor<float> weight_;  // [out_ch, in_ch, kernel_size, kernel_size]
    Tensor<float> bias_;    // [1, out_ch]
    size_type stride_, padding_, kernel_size_;
};

class MaxPool2d : public Model<MaxPool2d> {
public:
    ~MaxPool2d();
    MaxPool2d(size_type kernel_size, size_type stride = 0);  // stride=0 means stride=kernel_size

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer);

    std::vector<internal::Tensor*> parameters() const {
        return {};  // MaxPool2d has no parameters
    }

private:
    size_type kernel_size_, stride_;
};

class Flatten : public Model<Flatten> {
public:
    ~Flatten();
    Flatten(int start_dim = 1);

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer);

    std::vector<internal::Tensor*> parameters() const {
        return {};  // Flatten has no parameters
    }

private:
    int start_dim_;
};

struct ReLU : public Model<ReLU> {
    ReLU() = default;
    
    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
};

struct Softmax : public Model<Softmax> {
    int axis;
    Softmax(int axis);

    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
};

struct LogSoftmax : public Model<LogSoftmax> {
    int axis;
    LogSoftmax(int axis);
    
    std::vector<internal::Tensor*> parameters()const {
        return {};
    }

    Tensor<float> forward(Tensor<float> input);
    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) { return; }
};

// DÜZELTME: Nokta karakterini kaldırdım ve layer_variant'ı sınıf içinde tanımladım
class Sequence : public Model<Sequence> {  // "." karakteri KALDIRILDI!
    using layer_variant = std::variant<
        Linear,
        Conv2d,  
        MaxPool2d,
        Flatten,    
        ReLU,
        Softmax,
        LogSoftmax
    >;

    public:

    template<class ... Layers>
    Sequence(Layers&& ... layers) {
        layers_ = { std::forward<Layers>(layers)... };
    }
  
    Tensor<float> forward(Tensor<float> input) {
        for (auto& layer : layers_) {
            input = std::visit([input](auto&& argument) { return argument.forward(input); }, layer);
        }
        return input;
    }

    std::vector<internal::Tensor*> parameters() const {

        std::vector<internal::Tensor*> parameter_list;
        for (auto& layer : layers_) {
            std::visit([&parameter_list](auto&& argument) {
                for(auto parameter : argument.parameters()) {
                    parameter_list.push_back(parameter);
                }
                
            }, layer);
        }

        return parameter_list;
    }

    void set_optimizer(std::shared_ptr<net::base::Optimizer> optimizer) {
        for (auto& layer : layers_) {
            std::visit([optimizer](auto&& argument) { argument.set_optimizer(optimizer); }, layer);
        }
    }

    private:
    std::vector<layer_variant> layers_;
};

} // namespace net