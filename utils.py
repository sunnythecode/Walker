import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import csv
import cv2
import torch

def log_loss(loss_dict, global_steps, episode_num, csv_path = None):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Global_Steps", "Episode", "Alpha_Loss", "Critic_Loss", "Actor_Loss", "Alpha"])
        writer.writerow([global_steps, episode_num, loss_dict["alpha_loss"], loss_dict["critic_loss"], loss_dict["actor_loss"], loss_dict["alpha"]])

def log_episode(episode_num, steps_in_episode, global_steps, total_reward, max_steps, csv_path=None):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Episode", "StepsInEp", "GlobalSteps", "TotalReward", "MaxSteps"])
        writer.writerow([episode_num, steps_in_episode, global_steps, total_reward, max_steps])

def preprocess_img(image, out_shape=(64, 64), device='cpu'):
    """
    Preprocess an RGB image for PyTorch Conv2d.
    Returns a tensor of shape (1, C, H, W) with float32 values in [0, 1].
    """
    # Resize and convert to float32
    out = cv2.resize(image, out_shape)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    
    # Convert to torch tensor, reorder channels to (C, H, W)
    tensor = torch.tensor(out, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # Add batch dimension: (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    return tensor