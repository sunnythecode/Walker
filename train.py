import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import time
from models import *
from buffer import memoryBuffer
import os


# Constants
ACTION_SPACE = 4
STATE_SPACE = 24
MAX_MEMORY_SIZE = 250000
EPISODES = 600
MAX_STEPS = 300  # Decreased for quick training
BATCH_SIZE = 100
DISCOUNT_GAMMA = 0.99
LEARNING_RATE = 0.001  # Increased learning rate
TAU_RATE = 0.05
WARMUP_EPISODES = 50  # Increased for better initial exploration
UPDATE_FREQ = 2  # Decreased to prevent over-fitting to recent experiences
TARGET_ENTROPY = -ACTION_SPACE # Recommended target
CONTINUE_TRAINING = True # Continue training of a previously saved checkpoint
CHECKPOINT_PATH = "model_reuse/sac_best_model.pt"

os.makedirs('saved_models', exist_ok=True)

# Prep devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment
env = gym.make('BipedalWalker-v3')

# Actor, Critic networks
actor = Actor(ACTION_SPACE, STATE_SPACE).to(device)
critic = Dual_Critic(STATE_SPACE, ACTION_SPACE).to(device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optim = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE * 0.5)

critic_target = Dual_Critic(ACTION_SPACE, STATE_SPACE).to(device)
critic_target.load_state_dict(critic.state_dict())

if CONTINUE_TRAINING:
    checkpoint = torch.load(CHECKPOINT_PATH)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
    actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    critic_optim.load_state_dict(checkpoint["Q1_optimizer_state_dict"])

# Automatic entropy tuning:
alpha = torch.tensor(0.0062, dtype=torch.float32)
log_alpha = torch.tensor(torch.log(alpha), requires_grad=True, device=device)
alpha_optim = torch.optim.Adam([log_alpha], lr=LEARNING_RATE)

# Replay Experiences
replayBuffer = memoryBuffer(MAX_MEMORY_SIZE, STATE_SPACE, ACTION_SPACE, device)

# Logging
rewards_history = []
avg_rewards_history = []
max_reward = -100000

Q1_loss_history = []
Q2_loss_history = []
actor_loss_history = []


# Training loop
for episode in range(0, EPISODES):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)  # Changed to float32

    done = False
    steps = 0
    reward_total = 0

    actor_loss_print = None
    Q1_loss_print = None
    Q2_loss_print = None

    while not(done) and steps < MAX_STEPS:
        if episode < WARMUP_EPISODES:
            # Random actions in warmup
            action = torch.tensor(env.action_space.sample(), dtype=torch.float32).to(device)
        else:
            # Sampled actions in training
            action, _, _ = actor.sample(state)


        new_state, reward, done, finish, info = env.step(action.detach().cpu().numpy().squeeze())
        new_state = torch.tensor(new_state, dtype=torch.float32).to(device)  # Changed to float32

        # Store memory with consistent dtype
        memory = torch.cat([
            state,
            action.detach(),
            torch.tensor([reward], dtype=torch.float32).to(device),
            new_state,
            torch.tensor([float(done)], dtype=torch.float32).to(device)
        ]).to(device)
        replayBuffer.upload_memory(memory)

        state = new_state  # Add state update
        reward_total += reward
        steps += 1

        # Training Condition
        if episode > WARMUP_EPISODES:
            batch = replayBuffer.sample_batch(BATCH_SIZE)

            firstState = batch[:, :STATE_SPACE]
            action = batch[:, STATE_SPACE:STATE_SPACE+ACTION_SPACE]
            reward = batch[:, STATE_SPACE+ACTION_SPACE:STATE_SPACE+ACTION_SPACE+1]
            secondState = batch[:, STATE_SPACE+ACTION_SPACE+1:-1]
            done_val = batch[:, -1:]

            # Compute Q-values with gradient tracking disabled
            with torch.no_grad():
                secondAction, prob_action, _ = actor.sample(secondState)
                Q1, Q2 = critic_target(torch.cat([secondState, secondAction], dim=1))
                minimum_Q = torch.min(Q1, Q2)
                targetQ = reward + DISCOUNT_GAMMA * (1 - done_val) * (minimum_Q - alpha * prob_action)
                #print(f"Target Q: min={targetQ.min().item()}, max={targetQ.max().item()}, mean={targetQ.mean().item()}")


            # Q-function updates
            curr_Q1, curr_Q2 = critic(torch.cat([firstState, action], dim=1))
            
            Q1_loss = nn.MSELoss()(curr_Q1, targetQ.detach())
            Q2_loss = nn.MSELoss()(curr_Q2, targetQ.detach())

            Q1_loss_print = Q1_loss.item()
            Q2_loss_print = Q2_loss.item()

            Q1_loss_history.append(Q1_loss_print)
            Q2_loss_history.append(Q2_loss_print)

            critic_loss = Q1_loss + Q2_loss

            critic_optim.zero_grad()
            critic_loss.backward() # Removed Gradient Clipping
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optim.step()

            # Update actor on a lesser frequency to avoid fitting for no reason
            if (steps % UPDATE_FREQ == 0):
                # Actor update
                new_action, log_prob, _ = actor.sample(firstState)
                Q1, Q2 = critic(torch.cat([firstState, new_action], dim=1))

                minimum_Q = torch.min(Q1, Q2)

                actor_loss = -(minimum_Q - alpha * log_prob).mean()
                actor_loss_print = actor_loss.item()
                actor_loss_history.append(actor_loss_print)

                actor_optim.zero_grad()
                actor_loss.backward() # Removed Gradient Clipping
                actor_optim.step()

                alpha_loss = -(log_alpha * (log_prob + TARGET_ENTROPY).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                alpha = log_alpha.exp()


            #print(f"Q1 Loss: {Q1_loss.item():.3f}, Q2 Loss: {Q2_loss.item():.3f}, Actor Loss: {actor_loss.item():.3f}")

            # Soft update of target networks
            with torch.no_grad():
                for theta, theta_target in zip(critic.parameters(), critic_target.parameters()):
                    theta_target.data.copy_(TAU_RATE * theta.data + (1 - TAU_RATE) * theta_target.data)


    # Store reward history
    rewards_history.append(reward_total)
    avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    avg_rewards_history.append(avg_reward)
    
    
    # Save models periodically and on best performance
    if (episode + 1) % 20 == 0 or reward_total > max_reward :
        checkpoint = {
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'critic_target_state_dict': critic_target.state_dict(),
            'actor_optimizer_state_dict': actor_optim.state_dict(),
            'Q1_optimizer_state_dict': critic_optim.state_dict(),
            'episode': episode,
            'reward': reward_total,
            'avg_reward': avg_reward
        }
        torch.save(checkpoint, f'saved_models/sac_checkpoint_episode_{episode}.pt')
        
        # Save best model separately
        if reward_total > max_reward:
            torch.save(checkpoint, 'saved_models/sac_best_model.pt')
            max_reward = reward_total
    

        
    print(f"Warmup: {episode < WARMUP_EPISODES}, Episode {episode}, Steps: {steps}, Reward: {reward_total:.2f}, Avg Reward: {avg_reward:.2f}, Alpha: {alpha.item()}, Q1 Loss:{Q1_loss_print}, Q2 Loss: {Q2_loss_print}, Buffer: {replayBuffer.size()}")

env.close()

plt.figure()
plt.plot(Q1_loss_history, label='Q1 Loss')
plt.plot(Q2_loss_history, label='Q2 Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Critic Losses')
plt.legend()
plt.show()

# Plot actor loss on a separate plot
plt.figure()
plt.plot(actor_loss_history, label='Actor Loss', color='orange')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Actor Loss')
plt.legend()
plt.show()