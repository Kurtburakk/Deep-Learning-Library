#include "../include/CaberNet.h"
#include <iostream>

int main() {
    std::cout << "=== CaberNet Basit Test ===" << std::endl;
    
    try {
        // 1. Basit Tensor testi
        std::cout << "\n1. Tensor Creation Test:" << std::endl;
        net::Tensor<float> x({2, 3}, net::requires_gradient::False);
        x.fill({1, 2, 3, 4, 5, 6});
        std::cout << "Tensor x: " << x << std::endl;
        
        // 2. Tensor operasyonları testi
        std::cout << "\n2. Tensor Operations Test:" << std::endl;
        net::Tensor<float> y({2, 3}, net::requires_gradient::False);
        y.fill({0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
        std::cout << "Tensor y: " << y << std::endl;
        
        // Toplama testi
        auto sum_result = x + y;
        sum_result.perform();
        std::cout << "x + y: " << sum_result << std::endl;
        
        // Çarpma testi
        auto mult_result = x * y;
        mult_result.perform();
        std::cout << "x * y: " << mult_result << std::endl;
        
        // 3. Matrix multiplication testi
        std::cout << "\n3. Matrix Multiplication Test:" << std::endl;
        net::Tensor<float> w({3, 2}, net::requires_gradient::True);
        w.fill({0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
        std::cout << "Tensor w: " << w << std::endl;
        
        auto matmul_result = net::matmul(x, w);
        matmul_result.perform();
        std::cout << "x @ w result: " << matmul_result << std::endl;
        
        // 4. Gradient test
        std::cout << "\n4. Gradient Test:" << std::endl;
        net::Tensor<float> grad_input({2, 2}, true);
        grad_input.fill({1.0, 2.0, 3.0, 4.0});
        
        auto relu_result = net::function::relu(grad_input);
        relu_result.perform();
        std::cout << "ReLU result: " << relu_result << std::endl;
        
        // Backward pass testi
        net::Tensor<float> grad_output({2, 2}, false);
        grad_output.fill(1.0);
        relu_result.backward(grad_output);
        
        std::cout << "Gradient after ReLU: " << grad_input.gradient() << std::endl;
        
        std::cout << "\n=== Basit Test Başarılı! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Hata: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}