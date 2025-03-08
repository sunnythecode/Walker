import gymnasium as gym
import torch
import numpy as np
from models import *
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load the trained model weights"""
    checkpoint = torch.load(model_path)
    
    # Initialize models
    actor = Actor(4, 24).to(torch.float32)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # Set to evaluation mode
    
    return actor

def evaluate_model(model_path, num_episodes=10, render=True):
    """Evaluate the trained model"""
    # Load model
    actor = load_model(model_path)
    
    # Create environment
    env = gym.make('BipedalWalker-v3', render_mode='human' if render else None)
    
    rewards = []
    steps_list = []
    

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0
    steps = 0
    mean_hist = []
    std_hist = []
    
    while not (done):
        # Get action from policy
        with torch.no_grad():
            action, log_prob, mean_stds = actor.sample(state)
            mean_hist.append(mean_stds[0][0])
            std_hist.append(mean_stds[1][0])
            if steps % 12 == 0:
                print(f"STD: {mean_stds[1]} MEAN: {mean_stds[0]} STEP {steps}")

            
            action = mean_stds[0].cpu().numpy().squeeze()
        
        # Take step in environment
        state, reward, done, _, _ = env.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        
        total_reward += reward
        steps += 1
        
        if render:
            env.render()

    plt.plot(mean_hist)
    plt.plot(std_hist)
    plt.show()
    
    rewards.append(total_reward)
    steps_list.append(steps)
    print(f"Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f} ± {np.std(steps_list):.2f}")
    print(f"Best Episode Reward: {max(rewards):.2f}")
    print(f"Worst Episode Reward: {min(rewards):.2f}")
    
    return rewards, steps_list

if __name__ == "__main__":
    # Use the best model for evaluation
    model_path = 'saved_models/sac_best_model.pt'
    rewards, steps = evaluate_model(model_path, num_episodes=5, render=True)