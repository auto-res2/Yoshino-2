#!/usr/bin/env python3
"""
This module implements the evaluation functions for HiFiT-LoRA experiments.
It contains the three main experiments and evaluation metrics for the hierarchical transformer framework.
"""

import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from train import hiFiTLoRA_model, baselineFiT_model, train_one_epoch, create_optimizers


def measure_inference_time(model, device, input_tensor, iterations=10):
    """
    Measure the average inference time for a model.
    
    Args:
        model: The model to evaluate
        device: Device to run inference on (cuda or cpu)
        input_tensor: Input tensor for the model
        iterations: Number of iterations to average over
        
    Returns:
        Average inference time in seconds
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    avg_time = (time.time() - start) / iterations
    return avg_time

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def experiment_convergence(device):
    """
    Run experiment 1: Convergence Speed and Stability.
    Compares training loss and diffusion steps between HiFiT-LoRA and baseline FiT.
    
    Args:
        device: Device to run the experiment on (cuda or cpu)
    """
    print("\n--- Experiment 1: Convergence Speed and Stability ---")
    dummy_data = torch.randn(100, 3, 256, 256)
    dummy_labels = torch.zeros(100)  # dummy labels
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    
    optimizer_hifi, optimizer_baseline, scheduler_hifi, scheduler_baseline = create_optimizers(
        model_hifi, model_baseline, lr=1e-4
    )

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
        diffusion_steps_hifi.extend([d for (_, d) in losses_hifi])
        diffusion_steps_baseline.extend([d for (_, d) in losses_baseline])
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

    loss_vals_hifi = [l for (l, _) in loss_history_hifi]
    loss_vals_baseline = [l for (l, _) in loss_history_baseline]
    plt.figure(figsize=(6,4))
    plt.plot(loss_vals_hifi, label="HiFiT-LoRA")
    plt.plot(loss_vals_baseline, label="Baseline FiT")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss Comparison")
    plt.savefig("training_loss.pdf", bbox_inches="tight")
    plt.close()
    print("Saved training loss plot as training_loss.pdf")
    
    avg_diff_steps_hifi = np.mean(diffusion_steps_hifi)
    avg_diff_steps_baseline = np.mean(diffusion_steps_baseline)
    print(f"Average diffusion steps (HiFiT-LoRA): {avg_diff_steps_hifi:.2f}")
    print(f"Average diffusion steps (Baseline FiT): {avg_diff_steps_baseline:.2f}")


def experiment_hierarchical_token(device):
    """
    Run experiment 2: Hierarchical Token Refinement for High-Resolution Generalization.
    Evaluates model quality on different image resolutions.
    
    Args:
        device: Device to run the experiment on (cuda or cpu)
    """
    print("\n--- Experiment 2: Hierarchical Token Refinement ---")
    
    def generate_dummy_data(size):
        return torch.randn(16, 3, size[0], size[1])
    
    dataset_sizes = [(256,256), (256,512), (256,768)]
    metrics_hifi = {}
    metrics_baseline = {}

    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    
    for size in dataset_sizes:
        print(f"\nEvaluating for image size: {size[0]}x{size[1]}")
        dummy_images = torch.randn(8, 3, size[0], size[1])  # Reduced batch size for T4 GPU
        model_hifi.eval()
        model_baseline.eval()
        with torch.no_grad():
            outputs_hifi = model_hifi(dummy_images)
            outputs_baseline = model_baseline(dummy_images)
        mse_hifi = nn.MSELoss()(outputs_hifi, dummy_images).item()
        mse_baseline = nn.MSELoss()(outputs_baseline, dummy_images).item()
        eps = 1e-6
        quality_hifi = 1.0 / (mse_hifi + eps)
        quality_baseline = 1.0 / (mse_baseline + eps)
        metrics_hifi[size] = quality_hifi
        metrics_baseline[size] = quality_baseline
        print(f"HiFiT-LoRA quality metric: {quality_hifi:.2f} (simulated)")
        print(f"Baseline FiT quality metric: {quality_baseline:.2f} (simulated)")

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


def experiment_parameter_efficiency(device):
    """
    Run experiment 3: LoRA-Enhanced Parameter Efficiency.
    Compares parameter count, inference time, and memory usage between models.
    
    Args:
        device: Device to run the experiment on (cuda or cpu)
    """
    print("\n--- Experiment 3: LoRA-Enhanced Parameter Efficiency ---")
    model_hifi = hiFiTLoRA_model().to(device)
    model_baseline = baselineFiT_model().to(device)
    
    params_hifi = count_parameters(model_hifi)
    params_baseline = count_parameters(model_baseline)
    print(f"Number of learnable parameters (HiFiT-LoRA): {params_hifi}")
    print(f"Number of learnable parameters (Baseline FiT): {params_baseline}")
    
    dummy_input = torch.randn(8, 3, 256, 256)  # Reduced batch size for T4 GPU
    
    infer_time_hifi = measure_inference_time(model_hifi, device, dummy_input, iterations=5)
    infer_time_baseline = measure_inference_time(model_baseline, device, dummy_input, iterations=5)
    print(f"Average inference time (HiFiT-LoRA): {infer_time_hifi:.4f} sec per forward pass")
    print(f"Average inference time (Baseline FiT): {infer_time_baseline:.4f} sec per forward pass")
    
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
        
        torch.cuda.empty_cache()
    else:
        peak_mem_hifi = peak_mem_baseline = None
        print("CUDA not available; skipping peak memory measurement.")

    fig, ax1 = plt.subplots(figsize=(6,4))
    index = np.arange(2)
    bar_width = 0.35
    params = [params_hifi, params_baseline]
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
