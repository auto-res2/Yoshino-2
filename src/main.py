#!/usr/bin/env python3
"""
This script implements three experiments comparing HiFiT-LoRA against a baseline FiT model.
Experiment 1: Convergence Speed and Stability (training loss and diffusion steps)
Experiment 2: Hierarchical Token Refinement for High-Resolution Generalization (dummy image generation and quality metrics)
Experiment 3: LoRA-Enhanced Parameter Efficiency (parameter count, inference speed, memory usage)
A test function is included that quickly runs each experiment (using dummy data) so that the code can be verified.
"""

import os
import json
import torch

from evaluate import (
    experiment_convergence,
    experiment_hierarchical_token,
    experiment_parameter_efficiency
)


def load_config(config_path="config/experiment_config.json"):
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    else:
        print(f"Configuration file {config_path} not found. Using default configuration.")
        return {
            "experiment_name": "HiFiT-LoRA",
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "output_dir": "data",
            "status_enum": "stopped"
        }


def run_all_experiments_test():
    """
    Test function that quickly executes each experiment to confirm correct execution.
    The test uses dummy data and runs only for a brief duration.
    """
    print("\n===== Running All Experiments Test =====")
    
    config = load_config()
    output_dir = config.get("output_dir", "data")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    experiment_convergence(device)
    experiment_hierarchical_token(device)
    experiment_parameter_efficiency(device)
    
    for pdf_file in ["training_loss.pdf", "hierarchical_token_refinement.pdf", "efficiency_comparison.pdf"]:
        if os.path.exists(pdf_file) and not os.path.exists(os.path.join(output_dir, pdf_file)):
            os.rename(pdf_file, os.path.join(output_dir, pdf_file))
            print(f"Moved {pdf_file} to {output_dir}/ directory")
    
    print("All experiments executed successfully.")


if __name__ == "__main__":
    run_all_experiments_test()
    
    status_enum = "stopped"
    print(f"Experiment completed. Status: {status_enum}")
