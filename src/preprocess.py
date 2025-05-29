#!/usr/bin/env python3
"""
This module implements data preprocessing functions for HiFiT-LoRA experiments.
It contains utilities for generating dummy data and transformations for the experiments.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def create_dummy_dataset(num_samples=100, image_size=(256, 256), batch_size=8):
    """
    Create a dummy dataset with random images.
    
    Args:
        num_samples: Number of samples in the dataset
        image_size: Tuple of (height, width) for the images
        batch_size: Batch size for the dataloader
        
    Returns:
        DataLoader object with the dummy dataset
    """
    dummy_data = torch.randn(num_samples, 3, image_size[0], image_size[1])
    dummy_labels = torch.zeros(num_samples)
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_multi_resolution_datasets(batch_size=8):
    """
    Create datasets with multiple resolutions for hierarchical token experiments.
    
    Args:
        batch_size: Batch size for the dataloaders
        
    Returns:
        Dictionary mapping resolution tuples to DataLoader objects
    """
    resolutions = [(256, 256), (256, 512), (256, 768)]
    dataloaders = {}
    
    for res in resolutions:
        dummy_data = torch.randn(16, 3, res[0], res[1])
        dummy_labels = torch.zeros(16)
        dataset = TensorDataset(dummy_data, dummy_labels)
        dataloaders[res] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloaders

def generate_dummy_batch(batch_size=8, image_size=(256, 256), device='cpu'):
    """
    Generate a single batch of dummy data directly on the specified device.
    Useful for memory-efficient testing on GPU.
    
    Args:
        batch_size: Number of samples in the batch
        image_size: Tuple of (height, width) for the images
        device: Device to create the tensor on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (images, labels) tensors
    """
    images = torch.randn(batch_size, 3, image_size[0], image_size[1], device=device)
    labels = torch.zeros(batch_size, device=device)
    return images, labels

def optimize_for_t4_gpu(batch_size=8, max_resolution=(256, 768)):
    """
    Optimize data preprocessing parameters for NVIDIA Tesla T4 GPU with 16GB VRAM.
    
    Args:
        batch_size: Current batch size
        max_resolution: Maximum image resolution to be used
        
    Returns:
        Tuple of (optimized_batch_size, should_use_mixed_precision)
    """
    height, width = max_resolution
    memory_per_batch = batch_size * 3 * height * width * 4
    
    available_memory = 8 * (1024**3)  # 8GB in bytes
    
    if memory_per_batch > available_memory:
        optimized_batch_size = max(1, int(available_memory / (3 * height * width * 4)))
        should_use_mixed_precision = True
    else:
        optimized_batch_size = batch_size
        should_use_mixed_precision = False
    
    return optimized_batch_size, should_use_mixed_precision

def enable_mixed_precision():
    """
    Enable mixed precision training for memory optimization on GPU.
    
    Returns:
        GradScaler object for mixed precision training
    """
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled for memory optimization")
        return scaler
    else:
        print("CUDA not available; mixed precision not enabled")
        return None
