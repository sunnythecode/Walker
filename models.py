import torch.nn as nn
import torch

class Dual_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dual_Critic, self).__init__()

        self.fc1_Q1 = nn.Linear(state_dim + action_dim, 512, dtype=torch.float32)
        self.fc2_Q1 = nn.Linear(512, 512, dtype=torch.float32)
        self.fc3_Q1 = nn.Linear(512, 1, dtype=torch.float32)

        self.fc1_Q2 = nn.Linear(state_dim + action_dim, 512, dtype=torch.float32)
        self.fc2_Q2 = nn.Linear(512, 512, dtype=torch.float32)
        self.fc3_Q2 = nn.Linear(512, 1, dtype=torch.float32)

    def forward(self, x):
        q1 = self.fc1_Q1(x)
        q1 = torch.relu(q1)
        q1 = self.fc2_Q1(q1)
        q1 = torch.relu(q1)

        q1 = self.fc3_Q1(q1)

        
        q2 = self.fc1_Q2(x)
        q2 = torch.relu(q2)
        q2 = self.fc2_Q2(q2)
        q2 = torch.relu(q2)

        q2 = self.fc3_Q2(q2)

        return q1, q2

    
class Actor(nn.Module):
    def __init__(self, action_space, state_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 512, dtype=torch.float32)
        self.fc2 = nn.Linear(512, 512, dtype=torch.float32)
        self.fc3 = nn.Linear(512, action_space, dtype=torch.float32)
        self.log_std = nn.Linear(512, action_space, dtype=torch.float32)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.log_std.weight)

    def sample(self, state):
        mean, log_std = self.forward(state)
        stds = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, stds)

        x = dist.rsample()
        y = torch.tanh(x) # Allows for expressiveness(Explanation in README)
        
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = y
        mean = torch.tanh(mean)

        return action, log_prob, (mean, stds)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
