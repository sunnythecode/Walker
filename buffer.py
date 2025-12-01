import torch
import numpy as np

class MemoryBuffer:
    def __init__(self, max_capacity, state_dim, action_dim, image_shape=None, device="cpu"):
        """
        Args:
            max_capacity (int): max buffer size
            state_dim (int): dimension of vector state (if any)
            action_dim (int): dimension of action
            image_shape (tuple): (C, H, W) if storing images, else None
        """
        self.max_capacity = max_capacity
        self.device = device
        self.roll_point = 0
        self.size_counter = 0

        # Vector states
        self.states = torch.zeros((max_capacity, state_dim), dtype=torch.float32, device=device) if state_dim > 0 else None
        self.next_states = torch.zeros((max_capacity, state_dim), dtype=torch.float32, device=device) if state_dim > 0 else None

        # Images
        self.images = torch.zeros((max_capacity, *image_shape), dtype=torch.float32, device=device) if image_shape else None
        self.next_images = torch.zeros((max_capacity, *image_shape), dtype=torch.float32, device=device) if image_shape else None

        # Actions, rewards, dones
        self.actions = torch.zeros((max_capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_capacity, 1), dtype=torch.float32, device=device)

    def upload_memory(self, state, action, reward, next_state, done, image=None, next_image=None):
        """Store one transition safely detached from any computation graph"""
        idx = self.roll_point

        if self.states is not None:
            self.states[idx] = state.detach() if isinstance(state, torch.Tensor) else state
            self.next_states[idx] = next_state.detach() if isinstance(next_state, torch.Tensor) else next_state

        if self.images is not None and image is not None:
            self.images[idx] = image.detach() if isinstance(image, torch.Tensor) else image
            self.next_images[idx] = next_image.detach() if isinstance(next_image, torch.Tensor) else next_image

        self.actions[idx] = action.detach() if isinstance(action, torch.Tensor) else action
        self.rewards[idx] = reward.detach() if isinstance(reward, torch.Tensor) else reward
        self.dones[idx] = done.detach() if isinstance(done, torch.Tensor) else done

        # Update counters
        self.roll_point = (self.roll_point + 1) % self.max_capacity
        self.size_counter = min(self.size_counter + 1, self.max_capacity)


    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size_counter, batch_size, replace=False)

        batch = {
            "state": self.states[idxs] if self.states is not None else None,
            "image": self.images[idxs] if self.images is not None else None,
            "action": self.actions[idxs],
            "reward": self.rewards[idxs],
            "next_state": self.next_states[idxs] if self.next_states is not None else None,
            "next_image": self.next_images[idxs] if self.next_images is not None else None,
            "done": self.dones[idxs]
        }
        return batch

    def size(self):
        return self.size_counter
