#!/usr/bin/env python3
"""
This module implements the model definitions and training functions for HiFiT-LoRA experiments.
It contains the model architectures and training utilities for the hierarchical transformer framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DummyLoRALayer(nn.Module):
    """
    A dummy LoRA-like linear adaptation layer.
    It acts like a simple low-rank adaptation: output = x @ A + (x @ B)*alpha,
    where A is fixed and B (learnable) is low-rank.
    """
    def __init__(self, in_features, out_features, rank=2, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=False)
        nn.init.eye_(self.A[:min(in_features, out_features)])
        self.B = nn.Parameter(torch.randn(in_features, rank))
        self.C = nn.Parameter(torch.randn(rank, out_features))
    
    def forward(self, x):
        return torch.matmul(x, self.A) + self.alpha * torch.matmul(torch.matmul(x, self.B), self.C)

class DummyBaselineNorm(nn.Module):
    """A simple normalization for the baseline model."""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
    
    def forward(self, x):
        return self.norm(x)

class DummyHiFiTLoRANorm(nn.Module):
    """A dummy LayerNorm that uses a LoRA-like adaptation module."""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.lora = DummyLoRALayer(num_features, num_features, rank=2, alpha=0.5)
    
    def forward(self, x):
        norm_x = self.norm(x)
        return norm_x + self.lora(norm_x)

class hiFiTLoRA_model(nn.Module):
    """
    A dummy HiFiT-LoRA model that simulates an autoencoder.
    In a real implementation, this would be a flexible vision transformer enhanced with LoRA.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten(2)
        self.project = nn.Conv1d(16, 16, kernel_size=1)
        self.norm = DummyHiFiTLoRANorm(16)
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_diffusion_steps=False):
        enc = self.encoder(x)                          # shape (B,16,H,W)
        B, C, H, W = enc.shape
        flat = self.flatten(enc)                       # shape (B, C, H*W)
        proj = self.project(flat)                      # shape (B, C, H*W)
        token_seq = proj.transpose(1,2)                # shape (B, H*W, C)
        normed = self.norm(token_seq)                  # apply LoRA normalized adaptation
        token_seq = normed.transpose(1,2)              # back (B, C, H*W)
        unflat = token_seq.reshape(B, C, H, W)         # shape (B, C, H, W)
        recon = self.decoder(unflat)
        diffusion_steps = int(np.random.randint(1, 5))
        if return_diffusion_steps:
            return recon, diffusion_steps
        else:
            return recon

class baselineFiT_model(nn.Module):
    """
    A dummy baseline FiT model that is similar to the HiFiT-LoRA model but uses full-parameter LayerNorm.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten(2)
        self.project = nn.Conv1d(16, 16, kernel_size=1)
        self.norm = DummyBaselineNorm(16)  # standard LayerNorm, no LoRA
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_diffusion_steps=False):
        enc = self.encoder(x)
        B, C, H, W = enc.shape
        flat = self.flatten(enc)
        proj = self.project(flat)
        token_seq = proj.transpose(1,2)
        normed = self.norm(token_seq)
        token_seq = normed.transpose(1,2)
        unflat = token_seq.reshape(B, C, H, W)
        recon = self.decoder(unflat)
        diffusion_steps = int(np.random.randint(1, 5))
        if return_diffusion_steps:
            return recon, diffusion_steps
        else:
            return recon


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Train a model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader providing training data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device: Device to run training on (cuda or cpu)
        
    Returns:
        List of (loss, diffusion_steps) tuples for each batch
    """
    model.train()
    epoch_losses = []
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()
        output, diffusion_steps = model(images, return_diffusion_steps=True)
        loss = nn.MSELoss()(output, images)
        loss.backward()
        optimizer.step()
        scheduler.step()  # update scheduler (dummy update)
        epoch_losses.append((loss.item(), diffusion_steps))
    return epoch_losses

def create_optimizers(model_hifi, model_baseline, lr=1e-4):
    """
    Create optimizers and schedulers for both models.
    
    Args:
        model_hifi: The HiFiT-LoRA model
        model_baseline: The baseline FiT model
        lr: Learning rate
        
    Returns:
        Tuple of (optimizer_hifi, optimizer_baseline, scheduler_hifi, scheduler_baseline)
    """
    optimizer_hifi = optim.Adam(model_hifi.parameters(), lr=lr)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=lr)
    scheduler_hifi = optim.lr_scheduler.StepLR(optimizer_hifi, step_size=20, gamma=0.95)
    scheduler_baseline = optim.lr_scheduler.StepLR(optimizer_baseline, step_size=20, gamma=0.95)
    return optimizer_hifi, optimizer_baseline, scheduler_hifi, scheduler_baseline
