import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocess import HeterogeneousDataset, ImageTokenizer, custom_collate_fn
from train import BaselineDBC, CombinedTBDI, TBDIWithDiffusion, train_model
from evaluate import EvaluatedTBDI, evaluate_model, train_ablation

def experiment1():
    print("\n--- Experiment 1: Variable-Resolution and Heterogeneous State Representation ---")
    dataset = HeterogeneousDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    
    print("Training Baseline DBC Model:")
    baseline_model = BaselineDBC()
    optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    image_tokenizer = ImageTokenizer()
    loss_history_baseline = []
    for epoch in range(3):
        total_loss = 0.
        for batch in dataloader:
            images = batch['image']
            actions = batch['action']
            optimizer_baseline.zero_grad()
            
            all_tokens = []
            for img in images:
                tokens = image_tokenizer(img.unsqueeze(0))  # Add batch dimension
                all_tokens.append(tokens)
            
            pred_action = baseline_model(all_tokens)
            loss = nn.MSELoss()(pred_action, actions)
            loss.backward()
            optimizer_baseline.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history_baseline.append(avg_loss)
        print(f"Baseline Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    print("Training TBDI Model:")
    tbdi_model = CombinedTBDI()
    optimizer_tbdi = torch.optim.Adam(tbdi_model.parameters(), lr=1e-3)
    loss_history_tbdi = train_model(tbdi_model, dataloader, optimizer_tbdi, epochs=3)
    
    plt.figure()
    plt.plot(loss_history_baseline, label="Baseline DBC")
    plt.plot(loss_history_tbdi, label="TBDI")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison (Exp1)")
    plt.legend()
    plt.savefig("training_loss_exp1.pdf", bbox_inches="tight")
    plt.close()
    print("Experiment 1 plot saved as training_loss_exp1.pdf")

def experiment2():
    print("\n--- Experiment 2: Robustness Under OOD Conditions ---")
    dataset = HeterogeneousDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    
    tbdi_eval_model = EvaluatedTBDI()
    optimizer_tbdi = torch.optim.Adam(tbdi_eval_model.parameters(), lr=1e-3)
    print("Training TBDI Model (for OOD evaluation):")
    train_model(tbdi_eval_model, dataloader, optimizer_tbdi, epochs=3)
    
    loss_ood = evaluate_model(tbdi_eval_model, dataloader, drop_rate=0.2)
    
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    losses = []
    for rate in drop_rates:
        loss_val = evaluate_model(tbdi_eval_model, dataloader, drop_rate=rate)
        losses.append(loss_val)
    
    plt.figure()
    plt.plot(drop_rates, losses, marker='o')
    plt.xlabel("Drop Rate")
    plt.ylabel("Evaluation Loss (MSE)")
    plt.title("OOD Evaluation: Loss vs Token Drop Rate")
    plt.savefig("inference_latency_ood.pdf", bbox_inches="tight")
    plt.close()
    print("Experiment 2 plot saved as inference_latency_ood.pdf")

def experiment3():
    print("\n--- Experiment 3: Ablation Study on Loss Contributions ---")
    dataset = HeterogeneousDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    
    model_hybrid = TBDIWithDiffusion()
    optimizer_hybrid = torch.optim.Adam(model_hybrid.parameters(), lr=1e-3)
    
    print("Training with Hybrid Loss:")
    loss_history_hybrid = train_ablation(model_hybrid, dataloader, optimizer_hybrid, loss_mode="hybrid")
    
    print("Training with BC Loss Only:")
    loss_history_bc = train_ablation(model_hybrid, dataloader, optimizer_hybrid, loss_mode="bc")
    
    print("Training with Diffusion Loss Only:")
    loss_history_diff = train_ablation(model_hybrid, dataloader, optimizer_hybrid, loss_mode="diffusion")
    
    plt.figure()
    epochs_range = list(range(1, len(loss_history_hybrid)+1))
    plt.plot(epochs_range, loss_history_hybrid, label="Hybrid Loss")
    plt.plot(epochs_range, loss_history_bc, label="BC Loss Only")
    plt.plot(epochs_range, loss_history_diff, label="Diffusion Loss Only")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Ablation Study: Loss Dynamics")
    plt.legend()
    plt.savefig("training_loss_ablation.pdf", bbox_inches="tight")
    plt.close()
    print("Experiment 3 plot saved as training_loss_ablation.pdf")

def run_all_experiments_test():
    print("\n========== Running Self-Test of Experiments ==========")
    experiment1()
    experiment2()
    experiment3()
    print("\nSelf-Test Complete. All experiments ran and plots are saved as .pdf files.")

if __name__ == '__main__':
    run_all_experiments_test()
