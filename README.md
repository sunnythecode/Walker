# Reinforcement Learning with Soft Actor-Critic (SAC) for Bipedal Walking

Details for code listed below.

Video: https://youtu.be/OYJrw4SSvJ0

---

## **Key Components of the Code**

### **Constants and Initialization**

#### **Hyperparameters**:
- `ACTION_SPACE` and `STATE_SPACE`: Define the dimensions of the action and state spaces for the environment.
- `MAX_MEMORY_SIZE`: Size of the replay buffer to store experiences.
- `EPISODES`: Total training episodes.
- `MAX_STEPS`: Maximum steps per episode.
- `BATCH_SIZE`: Number of experiences sampled from the replay buffer per training step.
- `DISCOUNT_GAMMA`: Discount factor for future rewards.
- `LEARNING_RATE`, `TAU_RATE`, `WARMUP_STEPS`, `UPDATE_FREQ`: Hyperparameters for training, learning rate adjustments, and network updates.
- `TARGET_ENTROPY`: Encourages exploration by setting a target entropy value.

#### **Environment and Device**:
- Initializes the `BipedalWalker-v3` environment using Gymnasium.
- Determines whether to use GPU or CPU based on availability.

#### **Networks**:
- **Actor Network**: Outputs actions based on the current state.
- **Critic Networks**: Estimate Q-values for state-action pairs.
- **Target Critic**: Soft-updated version of the critic network for stable training.

#### **Replay Buffer**:
- Stores past experiences for sampling during training.
