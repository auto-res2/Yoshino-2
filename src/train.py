import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import ImageTokenizer, SensorTokenizer, drop_tokens

class BaselineDBC(nn.Module):
    def __init__(self, emb_dim=64, fixed_token_count=100):
        super(BaselineDBC, self).__init__()
        self.emb_dim = emb_dim
        self.fixed_token_count = fixed_token_count
        self.fc = nn.Linear(fixed_token_count * emb_dim, 3)

    def forward(self, tokens_list):
        padded_tokens = []
        for tokens in tokens_list:
            B, N, C = tokens.shape
            if N < self.fixed_token_count:
                pad = torch.zeros(B, self.fixed_token_count - N, C, device=tokens.device)
                tokens_padded = torch.cat([tokens, pad], dim=1)
            else:
                tokens_padded = tokens[:, :self.fixed_token_count, :]
            padded_tokens.append(tokens_padded)
        
        all_tokens = torch.cat(padded_tokens, dim=0)
        all_tokens = all_tokens.reshape(all_tokens.shape[0], -1)
        action_pred = self.fc(all_tokens)
        return action_pred

class TBDI(nn.Module):
    def __init__(self, emb_dim=64):
        super(TBDI, self).__init__()
        self.emb_dim = emb_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(emb_dim, 3)

    def forward(self, tokens):
        if isinstance(tokens, list):
            max_len = max(t.shape[1] for t in tokens)
            padded_tokens = []
            for t in tokens:
                B, N, C = t.shape
                if N < max_len:
                    pad = torch.zeros(B, max_len - N, C, device=t.device)
                    t_padded = torch.cat([t, pad], dim=1)
                else:
                    t_padded = t[:, :max_len, :]
                padded_tokens.append(t_padded)
            tokens = torch.cat(padded_tokens, dim=0)
        
        x = tokens.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        action_pred = self.fc(x)
        return action_pred

class CombinedTBDI(nn.Module):
    def __init__(self, patch_size=16, emb_dim=64):
        super(CombinedTBDI, self).__init__()
        self.image_tokenizer = ImageTokenizer(patch_size, emb_dim)
        self.sensor_tokenizer = SensorTokenizer(token_dim=16, emb_dim=emb_dim)
        self.aggregate_transformer = TBDI(emb_dim)

    def forward(self, images, sensors):
        image_tokens_list = []
        for img in images:
            tokens = self.image_tokenizer(img.unsqueeze(0))  # Add batch dimension
            image_tokens_list.append(tokens)
        
        sensor_tokens_list = []
        for sensor in sensors:
            sensor_tok = self.sensor_tokenizer(sensor)
            sensor_tokens_list.append(sensor_tok)
        
        combined_tokens_list = []
        for img_tok, sens_tok in zip(image_tokens_list, sensor_tokens_list):
            combined = torch.cat([img_tok, sens_tok], dim=1)
            combined_tokens_list.append(combined)
        
        action_pred = self.aggregate_transformer(combined_tokens_list)
        return action_pred

class EvaluatedTBDI(CombinedTBDI):
    def forward(self, images, sensors, drop_rate=0.2):
        image_tokens_list = []
        for img in images:
            tokens = self.image_tokenizer(img.unsqueeze(0))  # Add batch dimension
            image_tokens_list.append(tokens)
        
        sensor_tokens_list = []
        for sensor in sensors:
            sensor_tokens_list.append(self.sensor_tokenizer(sensor))
        
        combined_tokens_list = []
        for img_tok, sens_tok in zip(image_tokens_list, sensor_tokens_list):
            combined = torch.cat([img_tok, sens_tok], dim=1)
            combined = drop_tokens(combined, drop_rate)
            combined_tokens_list.append(combined)
        
        action_pred = self.aggregate_transformer(combined_tokens_list)
        return action_pred

class HybridLoss(nn.Module):
    def __init__(self, bc_weight=1.0, diffusion_weight=1.0):
        super(HybridLoss, self).__init__()
        self.bc_weight = bc_weight
        self.diffusion_weight = diffusion_weight
        self.bc_loss = nn.MSELoss()
        self.diffusion_loss = nn.MSELoss()

    def forward(self, pred_action, expert_action, diffusion_pred, diffusion_target):
        loss_bc = self.bc_loss(pred_action, expert_action)
        loss_diff = self.diffusion_loss(diffusion_pred, diffusion_target)
        total_loss = self.bc_weight * loss_bc + self.diffusion_weight * loss_diff
        return total_loss

class TBDIWithDiffusion(CombinedTBDI):
    def __init__(self, patch_size=16, emb_dim=64):
        super(TBDIWithDiffusion, self).__init__(patch_size, emb_dim)
        self.diffusion_head = nn.Linear(emb_dim, emb_dim)

    def forward(self, images, sensors):
        action_pred = super().forward(images, sensors)
        
        image_tokens_list = []
        for img in images:
            tokens = self.image_tokenizer(img.unsqueeze(0))  # Add batch dimension
            image_tokens_list.append(tokens)
        
        sensor_tokens_list = []
        for sensor in sensors:
            sensor_tokens_list.append(self.sensor_tokenizer(sensor))
        
        combined_tokens_list = []
        for img_tok, sens_tok in zip(image_tokens_list, sensor_tokens_list):
            combined = torch.cat([img_tok, sens_tok], dim=1)
            combined_tokens_list.append(combined)
        
        max_len = max(t.shape[1] for t in combined_tokens_list)
        padded_tokens = []
        for t in combined_tokens_list:
            B, N, C = t.shape
            if N < max_len:
                pad = torch.zeros(B, max_len - N, C, device=t.device)
                t_padded = torch.cat([t, pad], dim=1)
            else:
                t_padded = t[:, :max_len, :]
            padded_tokens.append(t_padded)
        tokens = torch.cat(padded_tokens, dim=0)
        
        x = tokens.transpose(0, 1)
        x = self.aggregate_transformer.transformer(x)
        aggregated = x.mean(dim=0)
        
        diffusion_pred = self.diffusion_head(aggregated)
        diffusion_target = image_tokens_list[0].mean(dim=1)
        return action_pred, diffusion_pred, diffusion_target

def train_model(model, dataloader, optimizer, epochs=5, loss_func=nn.MSELoss(), use_diffusion=False):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.
        for batch in dataloader:
            images = batch['image']
            sensors = batch['sensor']
            expert_action = batch['action']
            optimizer.zero_grad()
            if use_diffusion:
                pred_action, diffusion_pred, diffusion_target = model(images, sensors)
                loss = loss_func(pred_action, expert_action, diffusion_pred, diffusion_target)
            else:
                pred_action = model(images, sensors)
                loss = loss_func(pred_action, expert_action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    return loss_history
