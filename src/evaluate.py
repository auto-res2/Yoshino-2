import torch
import torch.nn as nn
from train import EvaluatedTBDI, TBDIWithDiffusion, HybridLoss

def evaluate_model(model, dataloader, drop_rate=0.2):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image']
            sensors = batch['sensor']
            expert_action = batch['action']
            pred_action = model(images, sensors, drop_rate)
            loss = criterion(pred_action, expert_action)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss (drop rate = {drop_rate}): {avg_loss:.4f}")
    return avg_loss

def train_ablation(model, dataloader, optimizer, loss_mode="hybrid", epochs=3):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.
        for batch in dataloader:
            images = batch['image']
            sensors = batch['sensor']
            expert_action = batch['action']
            optimizer.zero_grad()
            if loss_mode == "hybrid":
                criterion = HybridLoss(bc_weight=1.0, diffusion_weight=0.5)
                pred_action, diffusion_pred, diffusion_target = model(images, sensors)
                loss = criterion(pred_action, expert_action, diffusion_pred, diffusion_target)
            elif loss_mode == "bc":
                criterion_bc = nn.MSELoss()
                pred_action, _, _ = model(images, sensors)
                loss = criterion_bc(pred_action, expert_action)
            else:
                criterion_diff = nn.MSELoss()
                _, diffusion_pred, diffusion_target = model(images, sensors)
                loss = criterion_diff(diffusion_pred, diffusion_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} | Mode: {loss_mode} | Loss: {avg_loss:.4f}")
    return loss_history
