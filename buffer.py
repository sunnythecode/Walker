import torch
import numpy as np


class memoryBuffer():
    def __init__(self, max_capacity, state_space, action_space, device):
        self.max_capacity = max_capacity
        self.container = torch.tensor(np.zeros((1, state_space * 2 + action_space + 2)), dtype=torch.float32).to(device) # s, a, r, s', d
        self.roll_point = 0
        self.firstUpload = True
        self.device = device

    def upload_memory(self, memory):
        '''
        Input is a tensor that looks like [s, a, r, s', d]
        '''
        if memory.dim() == 1:
            memory = memory.view(1, memory.shape[0])
        if self.container.shape[0] < self.max_capacity:
            if self.firstUpload == True:
                self.container[0] = memory
                self.firstUpload = False
            else:
                self.container = torch.concat((self.container, memory)).to(self.device)
        else:
            self.container[self.roll_point] = memory
            self.roll_point = (self.roll_point + 1) % self.max_capacity
    
    def sample_batch(self, batch_size):
        return self.container[list(np.random.choice(self.container.shape[0], batch_size)), ].to(self.device)
    
    def size(self):
        return self.container.shape[0]
