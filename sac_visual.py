import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, input_shape = (64, 64), latent_dim=32):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # compute the flattened size after convolutions (assuming 84x84 input)
        self._conv_out_size = self._get_conv_out((input_channels, input_shape[0], input_shape[1]))
        print(self._conv_out_size)
        
        self.fc = nn.Linear(self._conv_out_size, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(torch.flatten(o, 1).shape[1])
    
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        returns: (batch_size, feature_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Dual_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size = 512):
        super(Dual_Critic, self).__init__()

        self.fc1_Q1 = nn.Linear(state_dim + action_dim, hidden_size, dtype=torch.float32)
        self.fc2_Q1 = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.fc3_Q1 = nn.Linear(hidden_size, 1, dtype=torch.float32)

        self.fc1_Q2 = nn.Linear(state_dim + action_dim, hidden_size, dtype=torch.float32)
        self.fc2_Q2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.fc3_Q2 = nn.Linear(hidden_size, 1, dtype=torch.float32)

        nn.init.xavier_uniform_(self.fc1_Q1.weight)
        nn.init.xavier_uniform_(self.fc2_Q1.weight)
        nn.init.xavier_uniform_(self.fc3_Q1.weight)
        nn.init.xavier_uniform_(self.fc1_Q2.weight)
        nn.init.xavier_uniform_(self.fc2_Q2.weight)
        nn.init.xavier_uniform_(self.fc3_Q2.weight)

    def forward(self, state, action):
        # concatenate state and action along last dimension
        x = torch.cat([state, action], dim=-1)

        # Q1 forward
        q1 = torch.relu(self.fc1_Q1(x))
        q1 = torch.relu(self.fc2_Q1(q1))
        q1 = self.fc3_Q1(q1)

        # Q2 forward
        q2 = torch.relu(self.fc1_Q2(x))
        q2 = torch.relu(self.fc2_Q2(q2))
        q2 = self.fc3_Q2(q2)

        return q1, q2
    
# Actor network
class Actor(nn.Module):
    def __init__(self, action_space, state_space, hidden_size = 512):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        self.fc3 = nn.Linear(hidden_size, action_space, dtype=torch.float32)
        self.log_std = nn.Linear(hidden_size, action_space, dtype=torch.float32)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.log_std.weight)

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        stds = torch.exp(log_std).clamp(min=1e-6, max=1e2)
        
        if deterministic:
            # Use the mean action for deterministic output
            x = mean
        else:
            dist = torch.distributions.Normal(mean, stds)
            x = dist.rsample()  # Reparameterized sample
        
        y = torch.tanh(x)  # Squash to [-1, 1]
        
        if deterministic:
            log_prob = None  # Not needed for deterministic actions
        else:
            log_prob = dist.log_prob(x)
            log_prob -= torch.log((1 - y.pow(2)).clamp(min=1e-6))
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = y
        mean_tanh = torch.tanh(mean)
        
        return action, log_prob, (mean_tanh, stds)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std


class SAC(nn.Module):
    def __init__(
        self,
        input_channels,
        input_shape,
        action_dim,
        state_dim,
        latent_dim=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        device='cuda'
    ):
        super(SAC, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Shared encoder
        self.encoder = CNNEncoder(input_channels, input_shape, latent_dim).to(device)
        self.encoder_target = CNNEncoder(input_channels, input_shape, latent_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # Actor
        self.actor = Actor(action_dim, state_dim + latent_dim).to(device)

        # Critic
        self.critic = Dual_Critic(state_dim + latent_dim, action_dim).to(device)
        self.critic_target = Dual_Critic(state_dim + latent_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -action_dim

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()), 
            lr=critic_lr, weight_decay=1e-4
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def alpha(self):
        return self.log_alpha.exp()

    # Sample action
    def sample_action(self, state, image, deterministic=False):
        latent = self.encoder(image)
        # breakpoint()
        return self.actor.sample(torch.cat([state, latent], dim = 1), deterministic)[0]

    # Critic forward
    def critic_forward(self, state, image, action):
        latent = self.encoder(image)
        return self.critic(torch.cat([state, latent], dim = 1), action)
    
    def save(self, filepath):
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'actor_fc': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu()
        }
        torch.save(save_dict, filepath)
        print(f"SAC agent saved to {filepath}")
    
    def load(self, filepath, map_location=None):
        checkpoint = torch.load(filepath, map_location=map_location)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor_fc'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = nn.Parameter(checkpoint['log_alpha'].to(self.device))
        print(f"SAC agent loaded from {filepath}")

    def update(self, replay_buffer, batch_size=256, actor_update=False, alpha_update = False):
        # Sample batch from replay buffer
        batch = replay_buffer.sample_batch(batch_size)

        device = self.device

        states = batch["state"].float().to(device) if batch["state"] is not None else None
        next_states = batch["next_state"].float().to(device) if batch["next_state"] is not None else None
        images = batch["image"].float().to(device) if batch["image"] is not None else None
        next_images = batch["next_image"].float().to(device) if batch["next_image"] is not None else None
        dones = batch["done"].float().to(device) if batch["done"] is not None else None
        actions = batch["action"].float().to(device) if batch["action"] is not None else None
        rewards = batch["reward"].float().to(device) if batch["reward"] is not None else None

        # --- Encode images ---
        latents = self.encoder(images)
        with torch.no_grad():
            next_latents = self.encoder_target(next_images)

        # Concatenate state and latent
        features = torch.cat([states, latents], dim=1)
        next_features = torch.cat([next_states, next_latents], dim=1)

        # --- Critic update ---
        with torch.no_grad():
            # Sample next action from actor
            next_actions, next_log_prob, _ = self.actor.sample(next_features, deterministic=False)
  
            # Target Q values
            q1_next, q2_next = self.critic_target(next_features, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha() * next_log_prob
            target_q = rewards + self.gamma * (1 - dones) * q_next

        # Current Q estimates
        q1, q2 = self.critic(features, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.encoder.parameters()), 
            max_norm=1.0
        )
        self.critic_optimizer.step()


        if actor_update:
            # actor update
            actor_features = features.detach()  
            sampled_actions, log_prob, _ = self.actor.sample(actor_features)
            q1_pi, q2_pi = self.critic(actor_features, sampled_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha() * log_prob - q_pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            if alpha_update:
                # --- Alpha update ---
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        

        # --- Soft update of target critic ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss':  actor_loss.item() if actor_update else None,
            'alpha_loss': alpha_loss.item() if (actor_update and alpha_update) else 0,
            'alpha': self.alpha().item()
        }
