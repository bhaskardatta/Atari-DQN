"""
DQN Agent implementation with experience replay and adaptive features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any
from copy import deepcopy

from .network import DQNNetwork, NoisyDQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, AdaptiveReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay.
    Supports standard DQN, Double DQN, and Dueling DQN variants.
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        batch_size: int = 32,
        buffer_size: int = 100000,
        min_buffer_size: int = 10000,
        device: str = "cuda",
        double_dqn: bool = True,
        dueling: bool = True,
        noisy_nets: bool = False,
        prioritized_replay: bool = False,
        adaptive_replay: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_shape: Shape of input states
            n_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Frequency of target network updates
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            min_buffer_size: Minimum buffer size before training
            device: Device to run on ('cuda' or 'cpu')
            double_dqn: Whether to use Double DQN
            dueling: Whether to use Dueling DQN
            noisy_nets: Whether to use noisy networks
            prioritized_replay: Whether to use prioritized replay
            adaptive_replay: Whether to use adaptive replay
            seed: Random seed for reproducibility
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.double_dqn = double_dqn
        self.noisy_nets = noisy_nets
        self.prioritized_replay = prioritized_replay
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize networks
        if noisy_nets:
            self.q_network = NoisyDQNNetwork(state_shape, n_actions, dueling=dueling).to(self.device)
            self.target_network = NoisyDQNNetwork(state_shape, n_actions, dueling=dueling).to(self.device)
        else:
            self.q_network = DQNNetwork(state_shape, n_actions, dueling=dueling).to(self.device)
            self.target_network = DQNNetwork(state_shape, n_actions, dueling=dueling).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if adaptive_replay:
            self.replay_buffer = AdaptiveReplayBuffer(buffer_size, seed=seed)
        elif prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, seed=seed)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        self.total_loss = 0.0
        self.loss_count = 0
        
        # Adaptation tracking
        self.performance_history = []
        self.adaptation_events = []
        self.environmental_changes = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy or noisy networks.
        
        Args:
            state: Current state
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        if self.noisy_nets:
            # Noisy networks handle exploration automatically
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, condition: Optional[int] = None):
        """Store experience in replay buffer."""
        if hasattr(self.replay_buffer, 'push') and condition is not None:
            # Adaptive replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done, condition)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None
        
        # Sample batch from replay buffer
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Prioritized replay returns 7 values
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            # Standard and adaptive replay return 5 values
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Calculate current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Calculate target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + \
                             (self.gamma * next_q_values * (~dones).unsqueeze(1))
        
        # Calculate loss
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Update priorities for prioritized replay
        if hasattr(self.replay_buffer, 'update_priorities') and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Optimize network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Reset noise for noisy networks
        if self.noisy_nets:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Update statistics
        self.total_loss += loss.item()
        self.loss_count += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """Update exploration rate."""
        if not self.noisy_nets:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, condition: Optional[int] = None) -> Optional[float]:
        """
        Perform one agent step: store experience and train.
        
        Returns:
            Training loss if training occurred
        """
        # Store experience
        self.store_experience(state, action, reward, next_state, done, condition)
        
        # Train if ready
        loss = self.train_step()
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Update exploration rate
        self.update_epsilon()
        
        self.step_count += 1
        
        return loss
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def detect_environment_change(self, recent_performance: list, window_size: int = 100) -> bool:
        """
        Detect if environment has changed based on performance degradation.
        
        Args:
            recent_performance: List of recent performance scores
            window_size: Size of window for comparison
        
        Returns:
            True if environment change detected
        """
        if len(recent_performance) < 2 * window_size:
            return False
        
        # Compare recent performance to previous performance
        recent_window = recent_performance[-window_size:]
        previous_window = recent_performance[-2*window_size:-window_size]
        
        recent_mean = np.mean(recent_window)
        previous_mean = np.mean(previous_window)
        
        # Detect significant performance drop
        performance_drop = (previous_mean - recent_mean) / (previous_mean + 1e-8)
        threshold = 0.2  # 20% performance drop
        
        change_detected = performance_drop > threshold
        
        if change_detected:
            self.environmental_changes.append({
                'step': self.step_count,
                'previous_performance': previous_mean,
                'current_performance': recent_mean,
                'performance_drop': performance_drop
            })
        
        return change_detected
    
    def adapt_to_change(self):
        """Adapt agent parameters when environment change is detected."""
        # Increase exploration temporarily
        if not self.noisy_nets:
            self.epsilon = min(0.5, self.epsilon * 2.0)
        
        # Reset target network update frequency
        self.target_update_freq = max(500, self.target_update_freq // 2)
        
        # Log adaptation event
        self.adaptation_events.append({
            'step': self.step_count,
            'new_epsilon': self.epsilon,
            'new_target_freq': self.target_update_freq
        })
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'performance_history': self.performance_history,
            'adaptation_events': self.adaptation_events,
            'environmental_changes': self.environmental_changes
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.performance_history = checkpoint.get('performance_history', [])
        self.adaptation_events = checkpoint.get('adaptation_events', [])
        self.environmental_changes = checkpoint.get('environmental_changes', [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        avg_loss = self.total_loss / max(1, self.loss_count)
        
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'avg_loss': avg_loss,
            'buffer_size': len(self.replay_buffer),
            'num_adaptations': len(self.adaptation_events),
            'num_env_changes': len(self.environmental_changes)
        }
