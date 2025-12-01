import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
import csv
from utils import *
#from sac_visual import *
from sac import *
from buffer import MemoryBuffer
from tqdm import tqdm
import math


# --- Config ---
ENV_NAME = 'BipedalWalker-v3'
# ENV_NAME = "BipedalWalkerHardcore-v3"
TRAIN_EPISODES = 4000        # how many training episodes
EVAL_INTERVAL = 10000       # evaluate every N steps
EVAL_EPISODES = 10        # number of eval episodes per evaluation


MODEL_LABEL = "base_blind" # name ur model here
SAVE_PATH = Path("models") / MODEL_LABEL
RESUME_TRAINING = False
WARMUP_STEPS = 2000

ACTION_DIM = 4
STATE_DIM = 24
MEMORY_CAPACITY = 200000
BATCH_SIZE = 512
# Critic
UPDATE_FREQ = 1 # Update every step
UPDATES_PER_STEP = 2 # How many critic updates every step


START_STEPS = 300 # Episode scaling - i recommend starting from 300-500
END_STEPS = 1600



# --- Environments ---
env = gym.make(ENV_NAME)     # training env
eval_env = gym.make(ENV_NAME)                     # eval env (no render)
os.makedirs(SAVE_PATH, exist_ok = True)

# Tracking
global_steps = 0
eval_steps = []
eval_rewards = []
episode_rewards = []  # per training episode

agent = SAC(3, (64, 64), ACTION_DIM, STATE_DIM)
buffer = MemoryBuffer(MEMORY_CAPACITY, STATE_DIM, ACTION_DIM, (3, 64, 64))

def get_max_steps(ep):
    progress = ep / TRAIN_EPISODES  # 0 → 1
    curve = progress ** 4          # quadratic curve
    return int(START_STEPS + curve * (END_STEPS - START_STEPS))

def evaluate_policy(env, episodes=5, episode_num=None, global_step=None, csv_path=None):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        # convert state to tensor on the agent's device
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        # image_obs = preprocess_img(env.render(), device=agent.device)  # already returns tensor
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():  # don't track gradients during evaluation
                action = agent.sample_action(state, deterministic=True)  # returns tensor

            # step the environment
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            # image_obs = preprocess_img(env.render(), device=agent.device)

            done = terminated or truncated
            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)

    # Ensure CSV exists and write header if new
    csv_file = Path(csv_path)
    file_exists = csv_file.exists()
    with csv_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["GlobalStep", "Episode", "AverageReward"])
        writer.writerow([global_step, episode_num, avg_reward])

    return avg_reward

env._max_episode_steps = START_STEPS

# train loop
for ep in range(TRAIN_EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)  # state tensor
    # image_obs = preprocess_img(env.render(), device=agent.device)           # image tensor
    done = False
    total_reward = 0
    steps_in_episode = 0
    with tqdm(total=1600, desc=f"Episode {ep+1}", unit="step") as pbar:
        # breakpoint()
        while not done:
            # breakpoint()
            if global_steps > WARMUP_STEPS:
                action = agent.sample_action(state)
            else:
                action = torch.tensor(env.action_space.sample()).to("cuda")


            
            prev_state = state.clone()

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy().flatten())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            done_tensor = torch.tensor([terminated or truncated], dtype=torch.float32, device=agent.device)

            # image_obs = preprocess_img(env.render(), device=agent.device)
            done = terminated or truncated

            # Update policy / buffer
            buffer.upload_memory(prev_state, action, reward_tensor, next_state, done_tensor)
            # print(buffer.size())
            if buffer.size() > 512 and global_steps % UPDATE_FREQ == 0:
                for update_num in range(UPDATES_PER_STEP):
                    actor_update = bool(update_num == 1)
                    loss_dict = agent.update(buffer, batch_size = BATCH_SIZE, actor_update=actor_update, alpha_update = global_steps > WARMUP_STEPS)
                # breakpoint()
                log_loss(loss_dict, global_steps, ep, SAVE_PATH / "loss.csv")

            total_reward += reward
            steps_in_episode += 1
            global_steps += 1
            state = next_state

            pbar.update(1)

            # Periodic evaluation
            if global_steps % EVAL_INTERVAL == 0:
                print("[EVAL]")
                avg_reward = evaluate_policy(eval_env, EVAL_EPISODES, ep, global_steps, SAVE_PATH / "val.csv")
                eval_steps.append(global_steps)
                eval_rewards.append(avg_reward)
                agent.save(SAVE_PATH / f"model_{global_steps}.pt")
                print(f"[Eval] Step {global_steps}: Avg Reward = {avg_reward:.2f}")

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1} finished in {steps_in_episode} steps, "
          f"total reward: {total_reward:.2f}")
    if global_steps < WARMUP_STEPS:
        print("[WARMUP] ON")
    env._max_episode_steps = get_max_steps(ep)
    print(f"[INFO] Max eps is {env._max_episode_steps}")
    print(f"[INFO] buffer size is {buffer.size()} / {MEMORY_CAPACITY} = {buffer.size() / MEMORY_CAPACITY}")
    log_episode(ep, steps_in_episode, global_steps, total_reward, env._max_episode_steps, SAVE_PATH / "train.csv")

env.close()
eval_env.close()



# plot stuff
plt.figure(figsize=(10,4))

# Training episode rewards
plt.subplot(1,2,1)
plt.plot(range(1, TRAIN_EPISODES+1), episode_rewards, marker='o')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards per Episode")

# Evaluation rewards
plt.subplot(1,2,2)
plt.plot(eval_steps, eval_rewards, marker='s', color='red')
plt.xlabel("Training Steps")
plt.ylabel("Average Eval Reward")
plt.title("Policy Evaluation During Training")

plt.tight_layout()
plt.savefig(SAVE_PATH)  # Save figure to file
plt.show()
