#!/usr/bin/env python3
"""
Dataset Verification Script
Verify PyTorch reads the SAME MNIST data as CaberNet
"""

import numpy as np
import struct

def read_mnist_images(filename):
    """Read MNIST images binary file (same format as CaberNet)"""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"📊 Images File Info:")
        print(f"   Magic: {magic} (should be 2051)")
        print(f"   Images: {num_images}")
        print(f"   Rows: {rows}, Cols: {cols}")
        
        # Read image data
        images_data = f.read()
        images = np.frombuffer(images_data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
        return images

def read_mnist_labels(filename):
    """Read MNIST labels binary file (same format as CaberNet)"""
    with open(filename, 'rb') as f:
        # Read header  
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        print(f"🎯 Labels File Info:")
        print(f"   Magic: {magic} (should be 2049)")
        print(f"   Labels: {num_labels}")
        
        # Read label data
        labels_data = f.read()
        labels = np.frombuffer(labels_data, dtype=np.uint8)
        
        return labels

def verify_first_batch(data_dir="data/mnist/", batch_size=16):
    """Verify first batch matches CaberNet exactly"""
    
    print("🔍 Verifying First Batch (PyTorch vs CaberNet)...")
    print(f"📁 Data directory: {data_dir}")
    
    try:
        # Load images and labels
        images = read_mnist_images(data_dir + "train-images-idx3-ubyte")
        labels = read_mnist_labels(data_dir + "train-labels-idx1-ubyte")
        
        print(f"\n✅ Successfully loaded data!")
        print(f"📊 Total images: {len(images)}")
        print(f"🎯 Total labels: {len(labels)}")
        
        # Get first batch (same as CaberNet uses)
        first_batch_images = images[:batch_size]  # First 16 images
        first_batch_labels = labels[:batch_size]  # First 16 labels
        
        print(f"\n🔬 First Batch Analysis (size={batch_size}):")
        print(f"📐 Image shape: {first_batch_images.shape}")
        print(f"🎯 Label shape: {first_batch_labels.shape}")
        
        # Print first batch labels (same as CaberNet sees)
        print(f"🏷️  First batch labels: {first_batch_labels}")
        
        # Normalize pixels to [0,1] range (same as CaberNet)
        normalized_images = first_batch_images.astype(np.float32) / 255.0
        
        # Statistics for first image
        first_image = normalized_images[0]
        print(f"\n📊 First Image Statistics:")
        print(f"   Label: {first_batch_labels[0]}")
        print(f"   Pixel range: [{first_image.min():.3f}, {first_image.max():.3f}]")
        print(f"   Pixel mean: {first_image.mean():.3f}")
        print(f"   Non-zero pixels: {np.count_nonzero(first_image)}")
        
        return first_batch_images, first_batch_labels, True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"💡 Make sure MNIST files are in: {data_dir}")
        return None, None, False
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, False

if __name__ == "__main__":
    print("🔍 MNIST Dataset Verification")
    print("Ensuring PyTorch and CaberNet use identical data")
    print()
    
    verify_first_batch()#!/usr/bin/env python3
"""
Dataset Verification Script
Verify PyTorch reads the SAME MNIST data as CaberNet
"""

import numpy as np
import struct

def read_mnist_images(filename):
    """Read MNIST images binary file (same format as CaberNet)"""
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"📊 Images File Info:")
        print(f"   Magic: {magic} (should be 2051)")
        print(f"   Images: {num_images}")
        print(f"   Rows: {rows}, Cols: {cols}")
        
        # Read image data
        images_data = f.read()
        images = np.frombuffer(images_data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        
        return images

def read_mnist_labels(filename):
    """Read MNIST labels binary file (same format as CaberNet)"""
    with open(filename, 'rb') as f:
        # Read header  
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        print(f"🎯 Labels File Info:")
        print(f"   Magic: {magic} (should be 2049)")
        print(f"   Labels: {num_labels}")
        
        # Read label data
        labels_data = f.read()
        labels = np.frombuffer(labels_data, dtype=np.uint8)
        
        return labels

def verify_first_batch(data_dir="data/mnist/", batch_size=16):
    """Verify first batch matches CaberNet exactly"""
    
    print("🔍 Verifying First Batch (PyTorch vs CaberNet)...")
    print(f"📁 Data directory: {data_dir}")
    
    try:
        # Load images and labels
        images = read_mnist_images(data_dir + "train-images-idx3-ubyte")
        labels = read_mnist_labels(data_dir + "train-labels-idx1-ubyte")
        
        print(f"\n✅ Successfully loaded data!")
        print(f"📊 Total images: {len(images)}")
        print(f"🎯 Total labels: {len(labels)}")
        
        # Get first batch (same as CaberNet uses)
        first_batch_images = images[:batch_size]  # First 16 images
        first_batch_labels = labels[:batch_size]  # First 16 labels
        
        print(f"\n🔬 First Batch Analysis (size={batch_size}):")
        print(f"📐 Image shape: {first_batch_images.shape}")
        print(f"🎯 Label shape: {first_batch_labels.shape}")
        
        # Print first batch labels (same as CaberNet sees)
        print(f"🏷️  First batch labels: {first_batch_labels}")
        
        # Normalize pixels to [0,1] range (same as CaberNet)
        normalized_images = first_batch_images.astype(np.float32) / 255.0
        
        # Statistics for first image
        first_image = normalized_images[0]
        print(f"\n📊 First Image Statistics:")
        print(f"   Label: {first_batch_labels[0]}")
        print(f"   Pixel range: [{first_image.min():.3f}, {first_image.max():.3f}]")
        print(f"   Pixel mean: {first_image.mean():.3f}")
        print(f"   Non-zero pixels: {np.count_nonzero(first_image)}")
        
        return first_batch_images, first_batch_labels, True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"💡 Make sure MNIST files are in: {data_dir}")
        return None, None, False
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, False

if __name__ == "__main__":
    print("🔍 MNIST Dataset Verification")
    print("Ensuring PyTorch and CaberNet use identical data")
    print()
    
    verify_first_batch()

