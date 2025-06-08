import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized inputs"""
    images = [item['image'] for item in batch]
    sensors = [item['sensor'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    return {'image': images, 'sensor': sensors, 'action': actions}

class HeterogeneousDataset(Dataset):
    def __init__(self, num_samples=100, image_resolutions=[(128,128), (160,120), (192,144)]):
        self.num_samples = num_samples
        self.image_resolutions = image_resolutions
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        res = random.choice(self.image_resolutions)
        img = Image.fromarray(np.uint8(np.random.rand(*res, 3) * 255))
        img = self.transform(img)
        sensor_length = random.randint(5, 10)
        sensor_data = torch.randn(sensor_length)
        expert_action = torch.randn(3)
        return {'image': img, 'sensor': sensor_data, 'action': expert_action}

class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=16, emb_dim=64):
        super(ImageTokenizer, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        tokens = self.proj(x)
        B, C, H, W = tokens.shape
        tokens = tokens.view(B, C, H * W).transpose(1, 2)
        return tokens

class SensorTokenizer(nn.Module):
    def __init__(self, token_dim=16, emb_dim=64):
        super(SensorTokenizer, self).__init__()
        self.linear = nn.Linear(token_dim, emb_dim)

    def forward(self, sensor_seq):
        L = sensor_seq.shape[0]
        if L >= 16:
            num_tokens = L // 16
            tokens = sensor_seq[: num_tokens * 16].view(num_tokens, 16)
        else:
            padded = F.pad(sensor_seq, (0, 16 - L))
            tokens = padded.unsqueeze(0)
        tokens = self.linear(tokens)
        return tokens.unsqueeze(0)

def drop_tokens(tokens, drop_rate=0.2):
    B, N, C = tokens.shape
    keep_mask = (torch.rand(B, N, device=tokens.device) > drop_rate).float()
    tokens_dropped = tokens * keep_mask.unsqueeze(-1)
    return tokens_dropped
