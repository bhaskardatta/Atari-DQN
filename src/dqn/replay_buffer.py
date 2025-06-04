"""
Experience replay buffer for DQN training.
"""

import random
import numpy as np
import torch
from collections import namedtuple, deque
from typing import List, Tuple, Optional


# Experience tuple
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Standard experience replay buffer for DQN.
    Stores experiences and provides random sampling for training.
    """
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
        if seed is not None:
            random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains only {len(self.buffer)} experiences, "
                           f"but {batch_size} requested")
        
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.tensor(np.array([e.state for e in experiences]), 
                             dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], 
                              dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], 
                              dtype=torch.float32)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), 
                                  dtype=torch.float32)
        dones = torch.tensor([e.done for e in experiences], 
                            dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    Samples experiences based on their TD error priority.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (starts at beta, increases to 1)
            beta_increment: How much to increment beta per sample
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Sum tree for efficient priority sampling
        self.tree_ptr = 0
        self.sum_tree = np.zeros(2 * capacity - 1)
        self.min_tree = np.full(2 * capacity - 1, float('inf'))
        self.data = np.zeros(capacity, dtype=object)
        
        if seed is not None:
            np.random.seed(seed)
    
    def _update_tree(self, tree_idx: int, priority: float):
        """Update sum and min trees with new priority."""
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        
        # Propagate changes up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += change
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx + 1],
                self.min_tree[2 * tree_idx + 2]
            )
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.sum_tree):
            return idx
        
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience with maximum priority."""
        experience = Experience(state, action, reward, next_state, done)
        
        tree_idx = self.tree_ptr + self.capacity - 1
        self.data[self.tree_ptr] = experience
        self._update_tree(tree_idx, self.max_priority ** self.alpha)
        
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self) < batch_size:
            raise ValueError(f"Buffer contains only {len(self)} experiences, "
                           f"but {batch_size} requested")
        
        indices = []
        priorities = []
        
        # Sample based on priorities
        priority_segment = self.sum_tree[0] / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            
            s = np.random.uniform(a, b)
            tree_idx = self._retrieve(0, s)
            data_idx = tree_idx - self.capacity + 1
            
            indices.append(data_idx)
            priorities.append(self.sum_tree[tree_idx])
        
        # Calculate importance sampling weights
        sampling_probs = np.array(priorities) / self.sum_tree[0]
        weights = np.power(len(self) * sampling_probs, -self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Get experiences
        experiences = [self.data[idx] for idx in indices]
        
        states = torch.tensor(np.array([e.state for e in experiences]), 
                             dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], 
                              dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], 
                              dtype=torch.float32)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), 
                                  dtype=torch.float32)
        dones = torch.tensor([e.done for e in experiences], 
                            dtype=torch.bool)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            # Clip priority to avoid zero
            priority = max(priority, 1e-6)
            self.max_priority = max(self.max_priority, priority)
            
            tree_idx = idx + self.capacity - 1
            self._update_tree(tree_idx, priority ** self.alpha)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return min(self.tree_ptr + self.capacity, self.capacity) if self.tree_ptr < self.capacity else self.capacity
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self) >= min_size


class AdaptiveReplayBuffer(ReplayBuffer):
    """
    Adaptive replay buffer that can handle environmental changes.
    Maintains separate buffers for different environmental conditions.
    """
    
    def __init__(self, capacity: int, num_conditions: int = 4, 
                 seed: Optional[int] = None):
        """
        Initialize adaptive replay buffer.
        
        Args:
            capacity: Total capacity across all conditions
            num_conditions: Number of different environmental conditions
            seed: Random seed for reproducibility
        """
        super().__init__(capacity, seed)
        
        self.num_conditions = num_conditions
        self.condition_capacity = capacity // num_conditions
        
        # Separate buffers for each condition
        self.condition_buffers = {
            i: deque(maxlen=self.condition_capacity) 
            for i in range(num_conditions)
        }
        
        # Track current environmental condition
        self.current_condition = 0
    
    def set_condition(self, condition: int):
        """Set current environmental condition."""
        if condition < 0 or condition >= self.num_conditions:
            raise ValueError(f"Condition must be between 0 and {self.num_conditions-1}")
        self.current_condition = condition
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, condition: Optional[int] = None):
        """
        Add experience to appropriate condition buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            condition: Environmental condition (uses current if None)
        """
        if condition is None:
            condition = self.current_condition
            
        experience = Experience(state, action, reward, next_state, done)
        self.condition_buffers[condition].append(experience)
        
        # Also add to main buffer for backward compatibility
        super().push(state, action, reward, next_state, done)
    
    def sample_balanced(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample experiences balanced across all conditions.
        
        Args:
            batch_size: Total number of experiences to sample
        
        Returns:
            Balanced batch of experiences
        """
        # Calculate samples per condition
        samples_per_condition = batch_size // self.num_conditions
        remaining_samples = batch_size % self.num_conditions
        
        all_experiences = []
        
        for condition in range(self.num_conditions):
            condition_buffer = self.condition_buffers[condition]
            
            if len(condition_buffer) == 0:
                continue
                
            # Number of samples for this condition
            n_samples = samples_per_condition
            if condition < remaining_samples:
                n_samples += 1
            
            # Sample from condition buffer
            n_samples = min(n_samples, len(condition_buffer))
            if n_samples > 0:
                experiences = random.sample(list(condition_buffer), n_samples)
                all_experiences.extend(experiences)
        
        # If we don't have enough experiences, fill from main buffer
        if len(all_experiences) < batch_size:
            remaining = batch_size - len(all_experiences)
            additional = random.sample(list(self.buffer), 
                                     min(remaining, len(self.buffer)))
            all_experiences.extend(additional)
        
        # Convert to tensors
        states = torch.tensor(np.array([e.state for e in all_experiences]), 
                             dtype=torch.float32)
        actions = torch.tensor([e.action for e in all_experiences], 
                              dtype=torch.long)
        rewards = torch.tensor([e.reward for e in all_experiences], 
                              dtype=torch.float32)
        next_states = torch.tensor(np.array([e.next_state for e in all_experiences]), 
                                  dtype=torch.float32)
        dones = torch.tensor([e.done for e in all_experiences], 
                            dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def get_condition_stats(self) -> dict:
        """Get statistics for each condition buffer."""
        stats = {}
        for condition, buffer in self.condition_buffers.items():
            stats[condition] = {
                'size': len(buffer),
                'capacity': buffer.maxlen,
                'utilization': len(buffer) / buffer.maxlen if buffer.maxlen > 0 else 0
            }
        return stats
