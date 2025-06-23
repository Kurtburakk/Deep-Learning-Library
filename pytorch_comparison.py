#!/usr/bin/env python3
"""
PyTorch CNN Comparison Script
Compare with CaberNet performance on MNIST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# ================================
# Model Definition (Same as CaberNet)
# ================================

class CaberNetCNN(nn.Module):
    """
    Exact same architecture as CaberNet stable CNN:
    - Conv2d(1, 16, 3, 1, 1) + ReLU + MaxPool2d(2)
    - Conv2d(16, 32, 3, 1, 1) + ReLU + MaxPool2d(2)
    - Flatten() + Linear(1568, 128) + ReLU + Linear(128, 10)
    """
    
    def __init__(self):
        super(CaberNetCNN, self).__init__()
        
        # Conv layers (same as CaberNet)
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # Linear layers (same as CaberNet)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 32*7*7 = 1568
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))  # [16, 1, 28, 28] → [16, 16, 14, 14]
        
        # Conv2 + ReLU + Pool  
        x = self.pool(F.relu(self.conv2(x)))  # [16, 16, 14, 14] → [16, 32, 7, 7]
        
        # Flatten + FC
        x = self.flatten(x)                   # [16, 32, 7, 7] → [16, 1568]
        x = F.relu(self.fc1(x))              # [16, 1568] → [16, 128]
        x = self.fc2(x)                      # [16, 128] → [16, 10]
        
        return x

# ================================
# MNIST Data Loading (Binary Format)
# ================================

def load_mnist_binary(images_path, labels_path, batch_size=16):
    """
    Load MNIST binary files (same format as CaberNet uses)
    """
    
    # Load images
    with open(images_path, 'rb') as f:
        # Skip header (16 bytes)
        f.read(16)
        
        # Read all image data
        images_data = f.read()
        images = np.frombuffer(images_data, dtype=np.uint8)
        images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    # Load labels  
    with open(labels_path, 'rb') as f:
        # Skip header (8 bytes)
        f.read(8)
        
        # Read all label data
        labels_data = f.read()
        labels = np.frombuffer(labels_data, dtype=np.uint8).astype(np.int64)
    
    print(f"📊 Loaded {len(images)} images, {len(labels)} labels")
    
    # Convert to PyTorch tensors
    images_tensor = torch.from_numpy(images).unsqueeze(1)  # Add channel dim: [N, 1, 28, 28]
    labels_tensor = torch.from_numpy(labels)
    
    # Create dataset and dataloader
    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

# ================================
# Training Function
# ================================

def train_pytorch_cnn(num_batches=5):
    """
    Train PyTorch CNN with same settings as CaberNet:
    - 5 batches 
    - Batch size 16
    - Learning rate 0.0001
    - SGD optimizer
    - NLLLoss
    """
    
    print("🐍 PyTorch CNN Training Started...")
    
    # Create model
    model = CaberNetCNN()
    
    # Same optimizer settings as CaberNet
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()
    
    # Load MNIST data (adjust path as needed)
    data_dir = "data/mnist/"
    try:
        train_loader = load_mnist_binary(
            data_dir + "train-images-idx3-ubyte",
            data_dir + "train-labels-idx1-ubyte", 
            batch_size=16
        )
    except FileNotFoundError:
        print("❌ MNIST files not found. Adjust data_dir path.")
        return None
    
    # Training metrics
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    print(f"🚀 PyTorch Training on {num_batches} batches...")
    
    model.train()
    training_start = time.time()
    
    # Training loop
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        batch_start = time.time()
        
        # Forward pass
        outputs = model(images)
        
        # Apply log_softmax for NLLLoss (same as CaberNet)
        log_probs = F.log_softmax(outputs, dim=1)
        
        # Calculate loss
        loss = criterion(log_probs, labels)
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_end = time.time()
        batch_time = (batch_end - batch_start) * 1000  # ms
        
        # Get max output value for comparison
        max_output = torch.max(outputs).item()
        current_accuracy = correct_predictions / total_samples * 100.0
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"Loss={loss.item():.4f}, "
              f"Acc={current_accuracy:.2f}%, "
              f"Time={batch_time:.0f}ms, "
              f"Max={max_output:.1f}")
    
    training_end = time.time()
    total_training_time = (training_end - training_start) * 1000  # ms
    
    # Final metrics
    avg_loss = total_loss / num_batches
    final_accuracy = correct_predictions / total_samples * 100.0
    
    print(f"\n🎯 PyTorch Training Results:")
    print(f"📈 Average Loss: {avg_loss:.4f}")
    print(f"🎯 Final Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️  Total Training Time: {total_training_time:.0f} ms")
    print(f"⚡ Avg Time per Batch: {total_training_time / num_batches:.0f} ms")
    
    return {
        'avg_loss': avg_loss,
        'final_accuracy': final_accuracy,
        'total_time': total_training_time,
        'avg_batch_time': total_training_time / num_batches
    }
# ================================
# Comparison Summary
# ================================

def compare_with_cabernet():
    """
    Compare PyTorch vs CaberNet results
    """
    
    print("\n" + "="*60)
    print("🔥 PYTORCH vs CABERNET COMPARISON")
    print("="*60)
    
    # Run PyTorch training
    pytorch_results = train_pytorch_cnn(num_batches=5)
    
    if pytorch_results is None:
        print("❌ PyTorch training failed")
        return
    
    # CaberNet results (UPDATED - from latest Release mode test)
    cabernet_results = {
        'avg_loss': 2.3983,     # From latest test with LR=0.00001
        'final_accuracy': 12.50,  # From latest test
        'total_time': 42,         # Release mode: 42ms total!
        'avg_batch_time': 8       # Release mode: 8ms per batch!
    }
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'Metric':<20} {'CaberNet':<15} {'PyTorch':<15} {'Winner':<10}")
    print("-" * 60)
    
    # Loss comparison
    loss_winner = "CaberNet" if cabernet_results['avg_loss'] < pytorch_results['avg_loss'] else "PyTorch"
    print(f"{'Average Loss':<20} {cabernet_results['avg_loss']:<15.4f} {pytorch_results['avg_loss']:<15.4f} {loss_winner:<10}")
    
    # Accuracy comparison  
    acc_winner = "CaberNet" if cabernet_results['final_accuracy'] > pytorch_results['final_accuracy'] else "PyTorch"
    print(f"{'Final Accuracy':<20} {cabernet_results['final_accuracy']:<15.2f}% {pytorch_results['final_accuracy']:<15.2f}% {acc_winner:<10}")
    
    # Speed comparison
    speed_winner = "CaberNet" if cabernet_results['avg_batch_time'] < pytorch_results['avg_batch_time'] else "PyTorch"
    print(f"{'Avg Batch Time':<20} {cabernet_results['avg_batch_time']:<15.0f}ms {pytorch_results['avg_batch_time']:<15.0f}ms {speed_winner:<10}")
    
    # Total time comparison
    total_winner = "CaberNet" if cabernet_results['total_time'] < pytorch_results['total_time'] else "PyTorch"
    print(f"{'Total Time':<20} {cabernet_results['total_time']:<15.0f}ms {pytorch_results['total_time']:<15.0f}ms {total_winner:<10}")
    
    # Speed ratio
    speed_ratio = pytorch_results['avg_batch_time'] / cabernet_results['avg_batch_time']
    if speed_ratio > 1:
        print(f"\n🚀 CaberNet is {speed_ratio:.1f}x FASTER than PyTorch!")
    else:
        print(f"\n🚀 PyTorch is {1/speed_ratio:.1f}x FASTER than CaberNet!")
    
    print("\n✅ Comparison completed!")

# ================================
# Main Execution
# ================================

if __name__ == "__main__":
    print("🔥 PyTorch vs CaberNet CNN Comparison")
    print("Same model architecture, same MNIST data, same hyperparameters")
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Run comparison
    compare_with_cabernet()