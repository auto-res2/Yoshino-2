#!/usr/bin/env python3
"""
This script implements three experiments comparing HiFiT-LoRA against a baseline FiT model.
Experiment 1: Convergence Speed and Stability (training loss and diffusion steps)
Experiment 2: Hierarchical Token Refinement for High-Resolution Generalization (dummy image generation and quality metrics)
Experiment 3: LoRA-Enhanced Parameter Efficiency (parameter count, inference speed, memory usage)
A test function is included that quickly runs each experiment (using dummy data) so that the code can be verified.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, utils as tv_utils
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

# -------------------------
# Dummy Model Definitions
# -------------------------
# For demonstration, we use very simple autoencoder-like models.
# The HiFiT-LoRA model simulates parameter reduction by using a dummy “LoRA” branch in its normalization layers.

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
        # full weight (not learnable, simulated as fixed identity projection)
        self.A = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=False)
        nn.init.eye_(self.A[:min(in_features, out_features)])
        # low-rank adaptation
        self.B = nn.Parameter(torch.randn(in_features, rank))
        self.C = nn.Parameter(torch.randn(rank, out_features))
    
    def forward(self, x):
        # x shape: (batch, ..., in_features)
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
        # Downsample (simulate encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Bottleneck with dummy HiFiT-LoRA normalized layer
        self.flatten = nn.Flatten(2)
        self.project = nn.Conv1d(16, 16, kernel_size=1)
        self.norm = DummyHiFiTLoRANorm(16)
        # Upsample (simulate decoder)
        self.unflatten = nn.Unflatten(2, (16, 16))
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_diffusion_steps=False):
        # Simulate a forward pass with dummy diffusion steps count
        enc = self.encoder(x)                          # shape (B,16,H,W)
        # flatten spatial dims into one dimension
        B, C, H, W = enc.shape
        flat = self.flatten(enc)                       # shape (B, C, H*W)
        proj = self.project(flat)                      # shape (B, C, H*W)
        # transpose for norm layer (simulate token sequences)
        token_seq = proj.transpose(1,2)                # shape (B, H*W, C)
        normed = self.norm(token_seq)                  # apply LoRA normalized adaptation
        token_seq = normed.transpose(1,2)              # back (B, C, H*W)
        # unflatten tokens back to image shape
        unflat = self.unflatten(token_seq)             # shape (B, C, H, W)
        recon = self.decoder(unflat)
        # For simulation, set dummy diffusion steps to a random (small integer) value.
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
        self.unflatten = nn.Unflatten(2, (16, 16))
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
        unflat = self.unflatten(token_seq)
        recon = self.decoder(unflat)
        diffusion_steps = int(np.random.randint(1, 5))
        if return_diffusion_steps:
            return recon, diffusion_steps
        else:
            return recon

# -------------------------
# Helper Functions
# -------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    epoch_losses = []
    for batch in dataloader:
        # Here batch: (images, labels); we ignore labels
        images, _ = batch
        images = images.to(device)
        optimizer.zero_grad()
        # Forward pass with dummy diffusion steps tracking
        output, diffusion_steps = model(images, return_diffusion_steps=True)
        # Use Mean Squared Error loss for reconstruction
        loss = nn.MSELoss()(output, images)
        loss.backward()
        optimizer.step()
        scheduler.step()  # update scheduler (dummy update)
        # Append a tuple: (loss, diffusion_steps)
        epoch_losses.append((loss.item(), diffusion_steps))
    return epoch_losses

def measure_inference_time(model, device, input_tensor, iterations=10):
    model.eval()
    input_tensor = input_tensor.to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    avg_time = (time.time() - start) / iterations
    return avg_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# Experiment 1: Convergence Speed and Stability
# -------------------------

def experiment_convergence(device):
    print("\n--- Experiment 1: Convergence Speed and Stability ---")
    # Create a dummy dataset: 100 samples of 3x256x256 random images
    dummy_data = torch.randn(100, 3, 256, 256)
    dummy_labels = torch.zeros(100)  # dummy labels
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Instantiate models
    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    
    optimizer_hifi = optim.Adam(model_hifi.parameters(), lr=1e-4)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=1e-4)
    # Use a step LR scheduler that mimics a rectified flow scheduler
    scheduler_hifi = optim.lr_scheduler.StepLR(optimizer_hifi, step_size=20, gamma=0.95)
    scheduler_baseline = optim.lr_scheduler.StepLR(optimizer_baseline, step_size=20, gamma=0.95)

    num_epochs = 2  # reduced epochs for quick testing
    loss_history_hifi = []
    loss_history_baseline = []
    diffusion_steps_hifi = []
    diffusion_steps_baseline = []

    for epoch in range(num_epochs):
        losses_hifi = train_one_epoch(model_hifi, dataloader, optimizer_hifi, scheduler_hifi, device)
        losses_baseline = train_one_epoch(model_baseline, dataloader, optimizer_baseline, scheduler_baseline, device)
        loss_history_hifi.extend(losses_hifi)
        loss_history_baseline.extend(losses_baseline)
        # Extract diffusion steps for current epoch
        diffusion_steps_hifi.extend([d for (_, d) in losses_hifi])
        diffusion_steps_baseline.extend([d for (_, d) in losses_baseline])
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

    # Prepare training loss curves
    loss_vals_hifi = [l for (l, _) in loss_history_hifi]
    loss_vals_baseline = [l for (l, _) in loss_history_baseline]
    plt.figure(figsize=(6,4))
    plt.plot(loss_vals_hifi, label="HiFiT-LoRA")
    plt.plot(loss_vals_baseline, label="Baseline FiT")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss Comparison")
    # Save as PDF using prescribed filename format
    plt.savefig("training_loss.pdf", bbox_inches="tight")
    plt.close()
    print("Saved training loss plot as training_loss.pdf")
    
    # Additionally, print average diffusion steps
    avg_diff_steps_hifi = np.mean(diffusion_steps_hifi)
    avg_diff_steps_baseline = np.mean(diffusion_steps_baseline)
    print(f"Average diffusion steps (HiFiT-LoRA): {avg_diff_steps_hifi:.2f}")
    print(f"Average diffusion steps (Baseline FiT): {avg_diff_steps_baseline:.2f}")

# -------------------------
# Experiment 2: Hierarchical Token Refinement for High-Resolution Generalization
# -------------------------

def experiment_hierarchical_token(device):
    print("\n--- Experiment 2: Hierarchical Token Refinement ---")
    # For simplicity we simulate a two-stage generation pipeline.
    # Stage 1: Downsample input (simulate coarse token generation)
    # Stage 2: Upsample tokens to high-resolution (simulate refinement)
    
    # We create a dummy dataset with varying aspect ratios.
    # For testing, we generate three types: 256x256, 256x512, and 256x768 images.
    def generate_dummy_data(size):
        return torch.randn(16, 3, size[0], size[1])
    
    dataset_sizes = [(256,256), (256,512), (256,768)]
    metrics_hifi = {}
    metrics_baseline = {}

    # Instantiate models (reuse the ones defined above)
    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    # In a full experiment, these models would be trained; here we simply simulate inference.
    
    for size in dataset_sizes:
        print(f"\nEvaluating for image size: {size[0]}x{size[1]}")
        dummy_images = torch.randn(16, 3, size[0], size[1])
        model_hifi.eval()
        model_baseline.eval()
        with torch.no_grad():
            outputs_hifi = model_hifi(dummy_images)
            outputs_baseline = model_baseline(dummy_images)
        # Simulate metric computation (here lower MSE implies better quality, used to mimic FID etc.)
        mse_hifi = nn.MSELoss()(outputs_hifi, dummy_images).item()
        mse_baseline = nn.MSELoss()(outputs_baseline, dummy_images).item()
        # Fake “quality” score: we define quality = 1/(MSE+eps)
        eps = 1e-6
        quality_hifi = 1.0 / (mse_hifi + eps)
        quality_baseline = 1.0 / (mse_baseline + eps)
        metrics_hifi[size] = quality_hifi
        metrics_baseline[size] = quality_baseline
        print(f"HiFiT-LoRA quality metric: {quality_hifi:.2f} (simulated)")
        print(f"Baseline FiT quality metric: {quality_baseline:.2f} (simulated)")

    # Plotting the quality metrics for each image size
    sizes_str = [f"{s[0]}x{s[1]}" for s in dataset_sizes]
    q_hifi = [metrics_hifi[s] for s in dataset_sizes]
    q_baseline = [metrics_baseline[s] for s in dataset_sizes]
    x = np.arange(len(dataset_sizes))
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, q_hifi, width, label="HiFiT-LoRA")
    plt.bar(x + width/2, q_baseline, width, label="Baseline FiT")
    plt.xticks(x, sizes_str)
    plt.ylabel("Simulated Quality Metric (1/MSE)")
    plt.xlabel("Image Resolution")
    plt.title("Hierarchical Token Refinement Quality")
    plt.legend()
    plt.savefig("hierarchical_token_refinement.pdf", bbox_inches="tight")
    plt.close()
    print("Saved hierarchical token refinement quality plot as hierarchical_token_refinement.pdf")

# -------------------------
# Experiment 3: LoRA-Enhanced Parameter Efficiency
# -------------------------

def experiment_parameter_efficiency(device):
    print("\n--- Experiment 3: LoRA-Enhanced Parameter Efficiency ---")
    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    
    params_hifi = count_parameters(model_hifi)
    params_baseline = count_parameters(model_baseline)
    print(f"Number of learnable parameters (HiFiT-LoRA): {params_hifi}")
    print(f"Number of learnable parameters (Baseline FiT): {params_baseline}")
    
    # Create a dummy input tensor (batch size 16, 3x256x256)
    dummy_input = torch.randn(16, 3, 256, 256)
    
    infer_time_hifi = measure_inference_time(model_hifi, device, dummy_input, iterations=5)
    infer_time_baseline = measure_inference_time(model_baseline, device, dummy_input, iterations=5)
    print(f"Average inference time (HiFiT-LoRA): {infer_time_hifi:.4f} sec per forward pass")
    print(f"Average inference time (Baseline FiT): {infer_time_baseline:.4f} sec per forward pass")
    
    # Check peak CUDA memory if available (only if running on CUDA)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model_hifi(dummy_input.to(device))
        peak_mem_hifi = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model_baseline(dummy_input.to(device))
        peak_mem_baseline = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Peak CUDA memory (HiFiT-LoRA): {peak_mem_hifi:.2f} MB")
        print(f"Peak CUDA memory (Baseline FiT): {peak_mem_baseline:.2f} MB")
    else:
        peak_mem_hifi = peak_mem_baseline = None
        print("CUDA not available; skipping peak memory measurement.")

    # Plot a bar chart showing parameter count and inference time comparisons
    fig, ax1 = plt.subplots(figsize=(6,4))
    index = np.arange(2)
    bar_width = 0.35
    # Parameter counts
    params = [params_hifi, params_baseline]
    # Inference time (we scale time to milliseconds for plotting)
    times = [infer_time_hifi*1000, infer_time_baseline*1000]
    
    color = 'tab:blue'
    ax1.bar(index - bar_width/2, params, bar_width, color=color, label="Parameter Count")
    ax1.set_ylabel("Parameter Count", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.bar(index + bar_width/2, times, bar_width, color=color, label="Inference Time (ms)")
    ax2.set_ylabel("Inference Time (ms)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.xticks(index, ["HiFiT-LoRA", "Baseline FiT"])
    plt.title("Parameter Efficiency Comparison")
    fig.tight_layout()
    plt.savefig("efficiency_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved parameter efficiency comparison plot as efficiency_comparison.pdf")

# -------------------------
# Test Function
# -------------------------
def run_all_experiments_test():
    """
    Test function that quickly executes each experiment to confirm correct execution.
    The test uses dummy data and runs only for a brief duration.
    """
    print("\n===== Running All Experiments Test =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Experiment 1
    experiment_convergence(device)
    # Experiment 2
    experiment_hierarchical_token(device)
    # Experiment 3
    experiment_parameter_efficiency(device)
    print("All experiments executed successfully. Test finished immediately.")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    run_all_experiments_test()
    
    status_enum = "stopped"
    print(f"Experiment completed. Status: {status_enum}")
