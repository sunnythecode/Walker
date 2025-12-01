import torch
import numpy as np
import csv
from pathlib import Path
import gymnasium as gym
import os
#from sac_visual import *
from sac import *
from utils import *

# -------- CONFIG --------
ENV_NAME = 'BipedalWalkerHardcore-v3'        # replace with your environment
MODEL_PATH = "models/base_8_blind_300_HC/model_890000.pt"  # replace with your model pt
    # path to your saved SAC agent
VIDEO_DIR = "./videos6"     
CSV_PATH = "./eval_rewards6.csv"
NUM_EPISODES = 10
ACTION_DIM = 4
STATE_DIM = 24

os.makedirs(VIDEO_DIR, exist_ok=True)

# -------- LOAD ENV --------
env = gym.make(ENV_NAME, render_mode = "rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda x: True)

# -------- LOAD MODEL --------
# Replace this with your agent loading logic
agent = SAC(3, (64, 64), ACTION_DIM, STATE_DIM)
agent.load(MODEL_PATH)
agent.eval()  # evaluation mode

# -------- EVALUATE --------
rewards = []

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    # image_obs = preprocess_img(env.render(), device=agent.device)
    done = False
    total_reward = 0
    step = 0

    while not done:
        with torch.no_grad():
            # Replace sample_action with your agent's method
            action = agent.sample_action(state, deterministic=True)

        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
        next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        # image_obs = preprocess_img(env.render(), device=agent.device)

        done = terminated or truncated
        total_reward += reward
        state = next_state
        step += 1

    rewards.append(total_reward)
    print(f"Episode {ep+1}/{NUM_EPISODES} reward: {total_reward}")

# -------- LOG RESULTS --------
avg_reward = np.mean(rewards)
csv_file = Path(CSV_PATH)
file_exists = csv_file.exists()
with csv_file.open("a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Episode", "AverageReward"])
    for i, r in enumerate(rewards):
        writer.writerow([i+1, r])

env.close()
print(f"Saved {NUM_EPISODES} episodes in {VIDEO_DIR}")
print(f"Average reward: {avg_reward}")
