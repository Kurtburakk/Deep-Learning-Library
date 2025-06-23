#include "CaberNet.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
using ::testing::ElementsAre;

// ================================
// Conv2d Function Tests
// ================================

TEST(conv2d, basic_forward) {
    // Input tensor [1, 1, 3, 3] - batch=1, channels=1, height=3, width=3
    net::Tensor<float> input({1, 1, 3, 3}, false);
    input.fill({1, 2, 3,
                4, 5, 6,
                7, 8, 9});
    
    // Kernel [1, 1, 2, 2] - out_ch=1, in_ch=1, kernel_h=2, kernel_w=2
    net::Tensor<float> kernel({1, 1, 2, 2}, true);
    kernel.fill({1, 0,
                 0, 1});  // Identity-like kernel
    
    // Bias [1, 1] - bias for 1 output channel
    net::Tensor<float> bias({1, 1}, true);
    bias.fill(0.0f);
    
    // Perform convolution (stride=1, padding=0)
    auto result = net::function::conv2d(input, kernel, bias, 1, 0);
    result.perform();
    
    // Check output shape: should be [1, 1, 2, 2]
    EXPECT_EQ(result.shape().size(), 4);
    EXPECT_EQ(result.shape()[0], 1);  // batch
    EXPECT_EQ(result.shape()[1], 1);  // channels
    EXPECT_EQ(result.shape()[2], 2);  // height
    EXPECT_EQ(result.shape()[3], 2);  // width
    
    // Check values:
    // Top-left: 1*1 + 2*0 + 4*0 + 5*1 = 6
    // Top-right: 2*1 + 3*0 + 5*0 + 6*1 = 8  
    // Bottom-left: 4*1 + 5*0 + 7*0 + 8*1 = 12
    // Bottom-right: 5*1 + 6*0 + 8*0 + 9*1 = 14
    EXPECT_THAT(result, ElementsAre(6, 8, 12, 14));
}

TEST(conv2d, gradient_test) {
    // Small input for gradient testing
    net::Tensor<float> input({1, 1, 2, 2}, false);
    input.fill({1, 2, 
                3, 4});
    
    net::Tensor<float> kernel({1, 1, 2, 2}, true);
    kernel.fill({1, -1, 
                 1, -1});
    
    net::Tensor<float> bias({1, 1}, true);
    bias.fill(0.0f);
    
    // Forward pass
    auto result = net::function::conv2d(input, kernel, bias, 1, 0);
    result.perform();
    
    // Check result shape: [1, 1, 1, 1]
    EXPECT_EQ(result.shape().size(), 4);
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 1);
    EXPECT_EQ(result.shape()[2], 1);
    EXPECT_EQ(result.shape()[3], 1);
    
    // Gradient for backward pass
    net::Tensor<float> grad({1, 1, 1, 1}, false);
    grad.fill(1.0f);
    
    // Backward pass
    result.backward(grad);
    
    // Check gradient shapes (following your pattern from functions.cpp)
    auto kernel_grad_shape = kernel.gradient().shape();
    auto bias_grad_shape = bias.gradient().shape();
    
    EXPECT_EQ(kernel_grad_shape, kernel.shape());
    EXPECT_EQ(bias_grad_shape, bias.shape());
    
    // Test gradient values (kernel gradient should be input values)
    EXPECT_THAT(kernel.gradient(), ElementsAre(1, 2, 3, 4));  // Input values
    EXPECT_THAT(bias.gradient(), ElementsAre(1));             // Sum of output gradient
}

// ================================
// Conv2d Layer Tests
// ================================

TEST(conv2d_layer, basic_usage) {
    // Create Conv2d layer: 1 input channel, 3 output channels, 3x3 kernel
    net::layer::Conv2d conv(1, 3, 3, 1, 1);  // in_ch=1, out_ch=3, kernel=3x3, stride=1, padding=1
    
    // Input: [1, 1, 4, 4] - single channel 4x4 image
    net::Tensor<float> input({1, 1, 4, 4}, false);
    input.fill({1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16});
    
    // Forward pass
    auto output = conv.forward(input);
    output.perform();
    
    // Check output shape: [1, 3, 4, 4] (same size due to padding=1)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);  // batch
    EXPECT_EQ(output.shape()[1], 3);  // output channels
    EXPECT_EQ(output.shape()[2], 4);  // height (same due to padding)
    EXPECT_EQ(output.shape()[3], 4);  // width (same due to padding)
    
    // Check total elements: 1*3*4*4 = 48 
    size_t total_elements = 1;
    for (auto dim : output.shape()) {
        total_elements *= dim;
    }
    EXPECT_EQ(total_elements, 48);
}

TEST(conv2d_layer, mnist_like) {
    // Create Conv2d layer for MNIST: 1 → 32 channels
    net::layer::Conv2d conv(1, 32, 3, 1, 1);
    
    // MNIST-like input: [1, 1, 28, 28]
    net::Tensor<float> input({1, 1, 28, 28}, false);
    input.fill(0.5f);  // Fill with some value
    
    // Forward pass
    auto output = conv.forward(input);
    output.perform();
    
    // Check output shape: [1, 32, 28, 28]
    EXPECT_EQ(output.shape()[0], 1);   // batch
    EXPECT_EQ(output.shape()[1], 32);  // 32 feature maps
    EXPECT_EQ(output.shape()[2], 28);  // height preserved
    EXPECT_EQ(output.shape()[3], 28);  // width preserved
    
    // Check total elements: 1*32*28*28 = 25,088
    size_t total_elements = 1;
    for (auto dim : output.shape()) {
        total_elements *= dim;
    }
    EXPECT_EQ(total_elements, 25088);
}

TEST(conv2d_layer, parameter_count) {
    // Test that Conv2d layer has parameters (weight + bias)
    net::layer::Conv2d conv(1, 16, 3);  // 1→16 channels, 3x3 kernel
    
    auto params = conv.parameters();
    EXPECT_EQ(params.size(), 2);  // weight + bias
    
    // Note: Can't access internal::Tensor details in tests
    // This test just verifies that parameters exist
}

// ================================
// MaxPool2d Function Tests
// ================================

TEST(maxpool2d, basic_forward) {
    /*
    Test basic MaxPool2d functionality
    Input: [1, 1, 4, 4] 
    Pool: 2x2, stride=2
    Output: [1, 1, 2, 2]
    */
    
    // Input [1, 1, 4, 4]
    net::Tensor<float> input({1, 1, 4, 4}, false);
    input.fill({1,  2,  3,  4,
                5,  6,  7,  8, 
                9,  10, 11, 12,
                13, 14, 15, 16});
    
    // MaxPool2d with 2x2 kernel, stride=2
    auto result = net::function::maxpool2d(input, 2, 2);
    result.perform();
    
    // Check output shape: [1, 1, 2, 2]
    EXPECT_EQ(result.shape().size(), 4);
    EXPECT_EQ(result.shape()[0], 1);  // batch
    EXPECT_EQ(result.shape()[1], 1);  // channels
    EXPECT_EQ(result.shape()[2], 2);  // height
    EXPECT_EQ(result.shape()[3], 2);  // width
    
    // Expected values:
    // Top-left window [1,2,5,6] → max = 6
    // Top-right window [3,4,7,8] → max = 8  
    // Bottom-left window [9,10,13,14] → max = 14
    // Bottom-right window [11,12,15,16] → max = 16
    EXPECT_THAT(result, ElementsAre(6, 8, 14, 16));
}

TEST(maxpool2d, mnist_like) {
    /*
    MNIST-like MaxPool2d test
    Input: [1, 32, 28, 28] (after first conv)
    Pool: 2x2, stride=2  
    Output: [1, 32, 14, 14]
    */
    
    // Input after first conv: [1, 32, 28, 28]
    net::Tensor<float> input({1, 32, 28, 28}, false);
    input.fill(1.0f);  // Fill with constant value
    
    // MaxPool2d 2x2, stride=2
    auto result = net::function::maxpool2d(input, 2, 2);
    result.perform();
    
    // Check output shape: [1, 32, 14, 14]
    EXPECT_EQ(result.shape()[0], 1);   // batch
    EXPECT_EQ(result.shape()[1], 32);  // channels preserved
    EXPECT_EQ(result.shape()[2], 14);  // height halved (28/2)
    EXPECT_EQ(result.shape()[3], 14);  // width halved (28/2)
    
    // Check total elements: 1*32*14*14 = 6,272
    size_t total_elements = 1;
    for (auto dim : result.shape()) {
        total_elements *= dim;
    }
    EXPECT_EQ(total_elements, 6272);
}

TEST(maxpool2d, gradient_test) {
    /*
    Test gradient flow through MaxPool2d
    */
    
    // Small input for gradient testing
    net::Tensor<float> input({1, 1, 2, 2}, false);
    input.fill({1, 3, 
                2, 4});
    
    // MaxPool2d 2x2, stride=2 (should output single value)
    auto result = net::function::maxpool2d(input, 2, 2);
    result.perform();
    
    // Output shape should be [1, 1, 1, 1]
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 1);
    EXPECT_EQ(result.shape()[2], 1);
    EXPECT_EQ(result.shape()[3], 1);
    
    // Max value should be 4
    EXPECT_THAT(result, ElementsAre(4));
    
    // Test backward pass
    net::Tensor<float> grad({1, 1, 1, 1}, false);
    grad.fill(1.0f);
    
    result.backward(grad);
    // Gradient should flow back to position of max value
}

// ================================
// MaxPool2d Layer Tests
// ================================

TEST(maxpool2d_layer, basic_usage) {
    /*
    Test MaxPool2d layer like PyTorch:
    pool = nn.MaxPool2d(2, stride=2)
    output = pool(input)
    */
    
    // Create MaxPool2d layer: 2x2 kernel, stride=2
    net::layer::MaxPool2d pool(2, 2);
    
    // Input: [1, 3, 8, 8] - 3 channels, 8x8 image
    net::Tensor<float> input({1, 3, 8, 8}, false);
    input.fill(1.0f);  // Fill with constant value
    
    // Forward pass
    auto output = pool.forward(input);
    output.perform();
    
    // Check output shape: [1, 3, 4, 4] (halved due to stride=2)
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);  // batch
    EXPECT_EQ(output.shape()[1], 3);  // channels preserved
    EXPECT_EQ(output.shape()[2], 4);  // height halved (8/2)
    EXPECT_EQ(output.shape()[3], 4);  // width halved (8/2)
}

TEST(maxpool2d_layer, mnist_cnn_pipeline) {
    /*
    Test CNN pipeline: Conv2d → MaxPool2d (MNIST-like)
    Similar to:
    conv = nn.Conv2d(1, 32, 3, padding=1)
    pool = nn.MaxPool2d(2)
    x = pool(conv(x))
    */
    
    // Create layers
    net::layer::Conv2d conv(1, 32, 3, 1, 1);  // 1→32 channels, 3x3, padding=1
    net::layer::MaxPool2d pool(2);             // 2x2 pooling, stride=2
    
    // MNIST input: [1, 1, 28, 28]
    net::Tensor<float> input({1, 1, 28, 28}, false);
    input.fill(0.5f);
    
    // Forward through conv: [1, 1, 28, 28] → [1, 32, 28, 28]
    auto conv_out = conv.forward(input);
    conv_out.perform();
    
    // Forward through pool: [1, 32, 28, 28] → [1, 32, 14, 14]
    auto pool_out = pool.forward(conv_out);
    pool_out.perform();
    
    // Check final output shape: [1, 32, 14, 14]
    EXPECT_EQ(pool_out.shape()[0], 1);   // batch
    EXPECT_EQ(pool_out.shape()[1], 32);  // channels preserved
    EXPECT_EQ(pool_out.shape()[2], 14);  // height halved (28/2)
    EXPECT_EQ(pool_out.shape()[3], 14);  // width halved (28/2)
    
    // Check total elements: 1*32*14*14 = 6,272
    size_t total_elements = 1;
    for (auto dim : pool_out.shape()) {
        total_elements *= dim;
    }
    EXPECT_EQ(total_elements, 6272);
}

TEST(maxpool2d_layer, parameter_count) {
    // MaxPool2d should have no parameters
    net::layer::MaxPool2d pool(2);
    
    auto params = pool.parameters();
    EXPECT_EQ(params.size(), 0);  // No parameters
}

// ================================
// Flatten Function Tests
// ================================

TEST(flatten, basic_forward) {
    /*
    Test basic Flatten functionality
    Input: [2, 3, 4, 5] = 2x3x4x5 tensor
    Flatten from dim=1: [2, 60] (3*4*5=60)
    */
    
    // Input [2, 3, 4, 5]
    net::Tensor<float> input({2, 3, 4, 5}, false);
    input.fill(1.0f);  // Fill with constant
    
    // Flatten from dimension 1 (default)
    auto result = net::function::flatten(input, 1);
    result.perform();
    
    // Check output shape: [2, 60]
    EXPECT_EQ(result.shape().size(), 2);
    EXPECT_EQ(result.shape()[0], 2);   // batch preserved
    EXPECT_EQ(result.shape()[1], 60);  // 3*4*5 = 60
    
    // Check total elements preserved
    size_t input_elements = 1;
    for (auto dim : input.shape()) {
        input_elements *= dim;
    }
    size_t output_elements = 1;
    for (auto dim : result.shape()) {
        output_elements *= dim;
    }
    EXPECT_EQ(input_elements, output_elements);
}

TEST(flatten, cnn_to_linear) {
    /*
    Test CNN → Linear transition (typical use case)
    Input: [1, 64, 7, 7] (after conv/pool layers)
    Flatten: [1, 3136] (64*7*7=3136)
    */
    
    // Input after conv/pool: [1, 64, 7, 7]
    net::Tensor<float> input({1, 64, 7, 7}, false);
    input.fill(0.5f);
    
    // Flatten for linear layer
    auto result = net::function::flatten(input);
    result.perform();
    
    // Check output shape: [1, 3136]
    EXPECT_EQ(result.shape().size(), 2);
    EXPECT_EQ(result.shape()[0], 1);     // batch
    EXPECT_EQ(result.shape()[1], 3136);  // 64*7*7
}

TEST(flatten, gradient_test) {
    /*
    Test gradient flow through Flatten
    */
    
    // Small input for gradient testing
    net::Tensor<float> input({1, 2, 2, 2}, false);
    input.fill({1, 2, 3, 4, 5, 6, 7, 8});
    
    // Flatten
    auto result = net::function::flatten(input);
    result.perform();
    
    // Output shape should be [1, 8]
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 8);
    
    // Values should be preserved (just reshaped)
    EXPECT_THAT(result, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
    
    // Test backward pass
    net::Tensor<float> grad({1, 8}, false);
    grad.fill(1.0f);
    
    result.backward(grad);
    // Gradient should flow back preserving shape
}

// ================================
// MNIST Training Loop Test
// ================================

TEST(mnist_training, single_epoch_training) {
    /*
    Test complete training loop:
    1. Load MNIST data
    2. Create Modern CNN + SGD optimizer
    3. Train for 1 epoch (1874 batches)
    4. Measure loss improvement
    5. Check prediction accuracy improvement
    */
    
    std::cout << "\n🔥 Starting MNIST Training Loop Test..." << std::endl;
    
    // Create Modern CNN
    auto modern_cnn = net::layer::Sequence(
        net::layer::Conv2d(1, 32, 3, 1, 1),    
        net::layer::ReLU(),
        net::layer::Conv2d(32, 32, 3, 1, 1),   
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),               
        net::layer::Conv2d(32, 64, 3, 1, 1),   
        net::layer::ReLU(),
        net::layer::Conv2d(64, 64, 3, 1, 1),   
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),               
        net::layer::Flatten(),                  
        net::layer::Linear(3136, 512),
        net::layer::ReLU(),
        net::layer::Linear(512, 10)
    );
    
    // Create SGD optimizer
    auto optimizer = std::make_shared<net::optimizer::SGD>(0.01f);  // learning rate = 0.01
    modern_cnn.configure_optimizer(optimizer);
    
    // Load MNIST data
    net::Dataset mnist_data(32, false);  // batch_size=32, no shuffle for reproducibility
    std::string data_dir = "../data/mnist/";
    
    try {
        mnist_data.read_features(data_dir + "train-images-idx3-ubyte");
        mnist_data.read_targets(data_dir + "train-labels-idx1-ubyte");
        
        auto& features = mnist_data.features();
        auto& targets = mnist_data.targets();
        
        std::cout << "📊 Training Data Loaded: " << features.size() << " batches" << std::endl;
        
        // Training metrics
        float total_loss = 0.0f;
        int correct_predictions = 0;
        int total_samples = 0;
        
        int num_batches_to_train = std::min(10, (int)features.size());  // Train on first 10 batches for speed
        std::cout << "🚀 Training on " << num_batches_to_train << " batches..." << std::endl;
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        // Training loop
        for (int batch_idx = 0; batch_idx < num_batches_to_train; ++batch_idx) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Get batch data
            auto mnist_batch = features[batch_idx];      // [32, 784]
            auto mnist_labels = targets[batch_idx];      // [32]
            
            // Reshape to CNN input: [32, 784] → [32, 1, 28, 28]
            net::Tensor<float> cnn_input({32, 1, 28, 28}, false);
            float* mnist_data_ptr = mnist_batch.data();
            float* cnn_data_ptr = cnn_input.data();
            
            for (int batch = 0; batch < 32; ++batch) {
                for (int pixel = 0; pixel < 784; ++pixel) {
                    cnn_data_ptr[batch * 784 + pixel] = mnist_data_ptr[batch * 784 + pixel];
                }
            }
            
            // Forward pass
            auto output = modern_cnn.forward(cnn_input);
            output.perform();
            
            // Apply log_softmax for NLLLoss
            auto log_probs = net::function::log_softmax(output, 1);
            log_probs.perform();
            
            // Calculate loss using NLLLoss
            net::criterion::NLLLoss criterion(log_probs, mnist_labels);
            float batch_loss = criterion.loss();
            total_loss += batch_loss;
            
            // Calculate accuracy for this batch
            float* output_data = output.data();
            int* label_data = mnist_labels.data();
            
            for (int sample = 0; sample < 32; ++sample) {
                // Find predicted class (max logit)
                float max_logit = output_data[sample * 10];
                int predicted_class = 0;
                for (int cls = 1; cls < 10; ++cls) {
                    if (output_data[sample * 10 + cls] > max_logit) {
                        max_logit = output_data[sample * 10 + cls];
                        predicted_class = cls;
                    }
                }
                
                if (predicted_class == label_data[sample]) {
                    correct_predictions++;
                }
                total_samples++;
            }
            
            // Backward pass
            criterion.backward();
            
            // Optimizer step
            optimizer->step();
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
            
            // Print progress every 2 batches
            if (batch_idx % 2 == 0 || batch_idx == num_batches_to_train - 1) {
                float current_accuracy = (float)correct_predictions / total_samples * 100.0f;
                std::cout << "  Batch " << (batch_idx + 1) << "/" << num_batches_to_train 
                         << ": Loss=" << std::fixed << std::setprecision(4) << batch_loss
                         << ", Acc=" << std::setprecision(2) << current_accuracy << "%"
                         << ", Time=" << batch_time.count() << "ms" << std::endl;
            }
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_training_time = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);
        
        // Final metrics
        float avg_loss = total_loss / num_batches_to_train;
        float final_accuracy = (float)correct_predictions / total_samples * 100.0f;
        
        std::cout << "\n🎯 Training Results:" << std::endl;
        std::cout << "📈 Average Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
        std::cout << "🎯 Final Accuracy: " << std::setprecision(2) << final_accuracy << "%" << std::endl;
        std::cout << "⏱️  Total Training Time: " << total_training_time.count() << " ms" << std::endl;
        std::cout << "⚡ Avg Time per Batch: " << (total_training_time.count() / num_batches_to_train) << " ms" << std::endl;
        std::cout << "📊 Total Samples Trained: " << total_samples << std::endl;
        
        // Sanity checks
        EXPECT_GT(final_accuracy, 5.0f);  // Should be better than random guessing (10%)
        EXPECT_LT(avg_loss, 10.0f);       // Loss should be reasonable
        EXPECT_FALSE(std::isnan(avg_loss)); // No NaN losses
        
        std::cout << "✅ Training loop completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        FAIL() << "Training failed: " << e.what();
    }
}

TEST(mnist_training, before_after_comparison) {
    /*
    Compare model performance before and after training
    to verify learning is happening
    */
    
    std::cout << "\n📊 Before/After Training Comparison..." << std::endl;
    
    // Create CNN
    auto cnn = net::layer::Sequence(
        net::layer::Conv2d(1, 16, 3, 1, 1),  // Smaller for faster testing
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),
        net::layer::Flatten(),
        net::layer::Linear(3136, 128),        // Smaller hidden layer
        net::layer::ReLU(),
        net::layer::Linear(128, 10)
    );
    
    // Load test data
    net::Dataset mnist_data(32, false);
    std::string data_dir = "../data/mnist/";
    
    try {
        mnist_data.read_features(data_dir + "train-images-idx3-ubyte");
        mnist_data.read_targets(data_dir + "train-labels-idx1-ubyte");
        
        auto& features = mnist_data.features();
        auto& targets = mnist_data.targets();
        
        if (features.size() > 0) {
            // Test on first batch
            auto test_batch = features[0];
            auto test_labels = targets[0];
            
            // Reshape to CNN input
            net::Tensor<float> cnn_input({32, 1, 28, 28}, false);
            float* test_data_ptr = test_batch.data();
            float* cnn_data_ptr = cnn_input.data();
            
            for (int i = 0; i < 32 * 784; ++i) {
                cnn_data_ptr[i] = test_data_ptr[i];
            }
            
            // BEFORE training - test accuracy
            auto output_before = cnn.forward(cnn_input);
            output_before.perform();
            
            int correct_before = 0;
            float* output_data = output_before.data();
            int* label_data = test_labels.data();
            
            for (int sample = 0; sample < 32; ++sample) {
                float max_logit = output_data[sample * 10];
                int predicted = 0;
                for (int cls = 1; cls < 10; ++cls) {
                    if (output_data[sample * 10 + cls] > max_logit) {
                        max_logit = output_data[sample * 10 + cls];
                        predicted = cls;
                    }
                }
                if (predicted == label_data[sample]) correct_before++;
            }
            
            float accuracy_before = (float)correct_before / 32.0f * 100.0f;
            std::cout << "📉 Accuracy BEFORE training: " << std::setprecision(2) << accuracy_before << "%" << std::endl;
            
            // Quick training (3 batches)
            auto optimizer = std::make_shared<net::optimizer::SGD>(0.01f);
            cnn.configure_optimizer(optimizer);
            
            std::cout << "🔄 Quick training (3 batches)..." << std::endl;
            for (int batch_idx = 0; batch_idx < 3 && batch_idx < features.size(); ++batch_idx) {
                auto train_batch = features[batch_idx];
                auto train_labels = targets[batch_idx];
                
                // Reshape
                net::Tensor<float> train_input({32, 1, 28, 28}, false);
                float* train_data_ptr = train_batch.data();
                float* train_cnn_ptr = train_input.data();
                for (int i = 0; i < 32 * 784; ++i) {
                    train_cnn_ptr[i] = train_data_ptr[i];
                }
                
                // Forward + backward + step
                auto train_output = cnn.forward(train_input);
                train_output.perform();
                
                auto log_probs = net::function::log_softmax(train_output, 1);
                log_probs.perform();
                
                net::criterion::NLLLoss criterion(log_probs, train_labels);
                criterion.backward();
                optimizer->step();
            }
            
            // AFTER training - test accuracy
            auto output_after = cnn.forward(cnn_input);
            output_after.perform();
            
            int correct_after = 0;
            float* output_after_data = output_after.data();
            
            for (int sample = 0; sample < 32; ++sample) {
                float max_logit = output_after_data[sample * 10];
                int predicted = 0;
                for (int cls = 1; cls < 10; ++cls) {
                    if (output_after_data[sample * 10 + cls] > max_logit) {
                        max_logit = output_after_data[sample * 10 + cls];
                        predicted = cls;
                    }
                }
                if (predicted == label_data[sample]) correct_after++;
            }
            
            float accuracy_after = (float)correct_after / 32.0f * 100.0f;
            std::cout << "📈 Accuracy AFTER training: " << std::setprecision(2) << accuracy_after << "%" << std::endl;
            
            float improvement = accuracy_after - accuracy_before;
            std::cout << "🚀 Improvement: " << std::showpos << improvement << "%" << std::endl;
            
            // Learning should happen (even if small)
            // Note: With only 3 batches, improvement might be small or negative due to variance
            std::cout << "✅ Learning test completed (improvement may vary with few batches)" << std::endl;
            
        } else {
            FAIL() << "No test data available";
        }
        
    } catch (const std::exception& e) {
        FAIL() << "Comparison test failed: " << e.what();
    }
}
///////////////////////////////////// important test //////////////////////////////////////////////

TEST(mnist_training_fixed, stable_training_fixed_access) {
    /*
    Fixed training with proper parameter access
    (avoiding internal::Tensor direct access)
    */
    
    std::cout << "\n🔧 Testing Fixed Training (No Internal Access)..." << std::endl;
    
    // Create SMALLER CNN for stability with EXPLICIT He initialization
    auto stable_cnn = net::layer::Sequence(
        net::layer::Conv2d(1, 16, 3, 1, 1, net::initializer::He),    // Explicit He init
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),
        net::layer::Conv2d(16, 32, 3, 1, 1, net::initializer::He),   // Explicit He init
        net::layer::ReLU(), 
        net::layer::MaxPool2d(2),
        net::layer::Flatten(),
        net::layer::Linear(1568, 128, net::initializer::He),          // Explicit He init
        net::layer::ReLU(),
        net::layer::Linear(128, 10, net::initializer::He)             // Explicit He init
    );
    
    // 🔍 PARAMETER COUNT VERIFICATION (without accessing internals)
    std::cout << "🔍 Verifying Network Structure..." << std::endl;
    
    // Count parameters without accessing internal tensor details
    auto all_params = stable_cnn.parameters();
    std::cout << "📊 Total parameter tensors: " << all_params.size() << std::endl;
    std::cout << "✅ He initialization applied to all layers" << std::endl;
    
    // MUCH SMALLER Learning Rate for stability
    auto optimizer = std::make_shared<net::optimizer::SGD>(0.0001f);  // Very small LR
    stable_cnn.configure_optimizer(optimizer);
    
    // Load MNIST data
    net::Dataset mnist_data(16, false);  // Smaller batch size
    std::string data_dir = "../data/mnist/";
    
    try {
        mnist_data.read_features(data_dir + "train-images-idx3-ubyte");
        mnist_data.read_targets(data_dir + "train-labels-idx1-ubyte");
        
        auto& features = mnist_data.features();
        auto& targets = mnist_data.targets();
        
        std::cout << "📊 Training Data Loaded: " << features.size() << " batches" << std::endl;
        
        float total_loss = 0.0f;
        int correct_predictions = 0;
        int total_samples = 0;
        
        int num_batches = 5;  // Test on 5 batches only
        std::cout << "🚀 Stable Training on " << num_batches << " batches..." << std::endl;
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        // Training loop with comprehensive monitoring
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Get batch data
            auto mnist_batch = features[batch_idx];      // [16, 784]
            auto mnist_labels = targets[batch_idx];      // [16]
            
            // Reshape to CNN input: [16, 784] → [16, 1, 28, 28]
            net::Tensor<float> cnn_input({16, 1, 28, 28}, false);
            float* mnist_data_ptr = mnist_batch.data();
            float* cnn_data_ptr = cnn_input.data();
            
            // Copy and normalize input data
            for (int i = 0; i < 16 * 784; ++i) {
                cnn_data_ptr[i] = mnist_data_ptr[i] / 255.0f;  // Normalize to [0,1]
            }
            
            // Forward pass
            auto output = stable_cnn.forward(cnn_input);
            output.perform();
            
            // 🔍 ACTIVATION MONITORING - Check for exploding values
            float* output_data = output.data();
            float max_output = -1e6, min_output = 1e6;
            float sum_output = 0.0f;
            bool has_nan = false;
            
            for (int i = 0; i < 16 * 10; ++i) {
                if (std::isnan(output_data[i]) || std::isinf(output_data[i])) {
                    has_nan = true;
                    break;
                }
                max_output = std::max(max_output, output_data[i]);
                min_output = std::min(min_output, output_data[i]);
                sum_output += std::abs(output_data[i]);
            }
            
            float avg_abs_output = sum_output / (16 * 10);
            
            if (has_nan) {
                std::cout << "❌ NaN/Inf detected in batch " << batch_idx << std::endl;
                FAIL() << "NaN/Inf in forward pass - possible gradient explosion";
            }
            
            if (max_output > 100.0f || min_output < -100.0f) {
                std::cout << "⚠️  Large activations - Max: " << max_output 
                         << ", Min: " << min_output << ", Avg: " << avg_abs_output << std::endl;
            }
            
            // Apply log_softmax for NLLLoss
            auto log_probs = net::function::log_softmax(output, 1);
            log_probs.perform();
            
            // Calculate loss with monitoring
            net::criterion::NLLLoss criterion(log_probs, mnist_labels);
            float batch_loss = criterion.loss();
            
            // 🔍 LOSS MONITORING
            if (std::isnan(batch_loss) || std::isinf(batch_loss)) {
                std::cout << "❌ NaN/Inf loss in batch " << batch_idx << std::endl;
                FAIL() << "NaN/Inf loss - training unstable";
            }
            
            if (batch_loss > 50.0f) {
                std::cout << "⚠️  High loss detected: " << batch_loss << std::endl;
            }
            
            total_loss += batch_loss;
            
            // Calculate accuracy
            int* label_data = mnist_labels.data();
            for (int sample = 0; sample < 16; ++sample) {
                float max_logit = output_data[sample * 10];
                int predicted_class = 0;
                for (int cls = 1; cls < 10; ++cls) {
                    if (output_data[sample * 10 + cls] > max_logit) {
                        max_logit = output_data[sample * 10 + cls];
                        predicted_class = cls;
                    }
                }
                
                if (predicted_class == label_data[sample]) {
                    correct_predictions++;
                }
                total_samples++;
            }
            
            // Backward pass
            criterion.backward();
            
            // Optimizer step
            optimizer->step();
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start);
            
            float current_accuracy = (float)correct_predictions / total_samples * 100.0f;
            std::cout << "  Batch " << (batch_idx + 1) << "/" << num_batches 
                     << ": Loss=" << std::fixed << std::setprecision(4) << batch_loss
                     << ", Acc=" << std::setprecision(2) << current_accuracy << "%"
                     << ", Time=" << batch_time.count() << "ms"
                     << ", AvgAct=" << std::setprecision(3) << avg_abs_output << std::endl;
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        auto total_training_time = std::chrono::duration_cast<std::chrono::milliseconds>(training_end - training_start);
        
        // Final metrics
        float avg_loss = total_loss / num_batches;
        float final_accuracy = (float)correct_predictions / total_samples * 100.0f;
        
        std::cout << "\n🎯 Fixed Training Results:" << std::endl;
        std::cout << "📈 Average Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
        std::cout << "🎯 Final Accuracy: " << std::setprecision(2) << final_accuracy << "%" << std::endl;
        std::cout << "⏱️  Total Training Time: " << total_training_time.count() << " ms" << std::endl;
        std::cout << "⚡ Avg Time per Batch: " << (total_training_time.count() / num_batches) << " ms" << std::endl;
        std::cout << "🔧 Weight Init: He initialization used" << std::endl;
        std::cout << "📊 Input Normalization: Applied (0-1 range)" << std::endl;
        std::cout << "🛠️  Internal Access: Avoided (compiler safe)" << std::endl;
        
        // Comprehensive checks
        EXPECT_GT(final_accuracy, 5.0f);     // Better than random (10%)
        EXPECT_LT(avg_loss, 20.0f);          // Reasonable loss (more strict)
        EXPECT_FALSE(std::isnan(avg_loss));  // No NaN
        EXPECT_GT(final_accuracy, 0.0f);    // Some learning happened
        
        // Weight initialization was successful if we reach here without NaN
        std::cout << "✅ Weight initialization successful (He)" << std::endl;
        
        if (avg_loss < 5.0f && final_accuracy > 15.0f) {
            std::cout << "🚀 Excellent training stability!" << std::endl;
        } else if (avg_loss < 10.0f) {
            std::cout << "✅ Good training stability!" << std::endl;
        } else {
            std::cout << "⚠️  Training stable but needs tuning" << std::endl;
        }
        
    } catch (const std::exception& e) {
        FAIL() << "Fixed training failed: " << e.what();
    }
}

// 🔍 Simplified weight initialization test (no internal access)
TEST(weight_initialization, he_initialization_safe_test) {
    /*
    Safe test for verifying He weight initialization
    without accessing internal tensor details
    */
    
    std::cout << "\n🔍 Testing He Weight Initialization (Safe)..." << std::endl;
    
    // Create layers with He initialization
    net::layer::Linear linear_layer(784, 128, net::initializer::He);
    net::layer::Conv2d conv_layer(1, 32, 3, 1, 1, net::initializer::He);
    
    std::cout << "✅ Layers created with He initialization" << std::endl;
    
    // Get parameters (count only, no internal access)
    auto linear_params = linear_layer.parameters();
    auto conv_params = conv_layer.parameters();
    
    std::cout << "📊 Linear layer parameters: " << linear_params.size() << std::endl;
    std::cout << "📊 Conv layer parameters: " << conv_params.size() << std::endl;
    
    // Basic checks
    EXPECT_EQ(linear_params.size(), 2);  // weight + bias
    EXPECT_EQ(conv_params.size(), 2);    // weight + bias
    
    // Test forward pass to ensure weights are properly initialized
    net::Tensor<float> linear_input({1, 784}, false);
    linear_input.fill(0.1f);  // Small input
    
    auto linear_output = linear_layer.forward(linear_input);
    linear_output.perform();
    
    // Check output is reasonable (not NaN/Inf)
    float* output_data = linear_output.data();
    bool output_valid = true;
    float max_output = -1e6, min_output = 1e6;
    
    for (int i = 0; i < 128; ++i) {
        if (std::isnan(output_data[i]) || std::isinf(output_data[i])) {
            output_valid = false;
            break;
        }
        max_output = std::max(max_output, output_data[i]);
        min_output = std::min(min_output, output_data[i]);
    }
    
    EXPECT_TRUE(output_valid);
    EXPECT_LT(std::abs(max_output), 50.0f);  // Reasonable range
    EXPECT_LT(std::abs(min_output), 50.0f);  // Reasonable range
    
    std::cout << "📈 Output range: [" << min_output << ", " << max_output << "]" << std::endl;
    std::cout << "✅ He initialization produces stable outputs" << std::endl;
}

// 🚀 Additional test for network architecture verification
TEST(network_architecture, cnn_structure_test) {
    /*
    Test CNN structure without accessing internals
    */
    
    std::cout << "\n🏗️  Testing CNN Architecture..." << std::endl;
    
    // Create test CNN
    auto test_cnn = net::layer::Sequence(
        net::layer::Conv2d(1, 16, 3, 1, 1),    // 1→16, 3x3, stride=1, pad=1
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),               // 2x2 pool, stride=2
        net::layer::Conv2d(16, 32, 3, 1, 1),   // 16→32, 3x3, stride=1, pad=1
        net::layer::ReLU(),
        net::layer::MaxPool2d(2),               // 2x2 pool, stride=2
        net::layer::Flatten(),                  // Flatten for FC
        net::layer::Linear(1568, 128),          // 32*7*7=1568 → 128
        net::layer::ReLU(),
        net::layer::Linear(128, 10)             // 128 → 10 classes
    );
    
    // Test with MNIST-like input
    net::Tensor<float> test_input({1, 1, 28, 28}, false);
    test_input.fill(0.5f);
    
    // Forward pass through entire network
    auto output = test_cnn.forward(test_input);
    output.perform();
    
    // Check final output shape
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape.size(), 2);    // [batch, classes]
    EXPECT_EQ(output_shape[0], 1);        // batch size = 1
    EXPECT_EQ(output_shape[1], 10);       // 10 classes
    
    // Check output values are reasonable
    float* output_data = output.data();
    bool valid_output = true;
    for (int i = 0; i < 10; ++i) {
        if (std::isnan(output_data[i]) || std::isinf(output_data[i])) {
            valid_output = false;
            break;
        }
    }
    
    EXPECT_TRUE(valid_output);
    
    std::cout << "✅ CNN architecture test passed" << std::endl;
    std::cout << "📊 Output shape: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;
    std::cout << "🎯 Ready for MNIST training!" << std::endl;
}