"""
Reference-compatible DQN network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ReferenceDQNNetwork(nn.Module):
    """
    DQN Network compatible with reference implementation.
    Handles input format (H, W, C) like the reference code.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (84, 84, 4),  # H, W, C format
        n_actions: int = 4,
        initialize_weights: bool = False
    ):
        """
        Initialize DQN network in reference format.
        
        Args:
            input_shape: (height, width, channels) - reference format
            n_actions: Number of actions
            initialize_weights: Whether to manually initialize weights
        """
        super(ReferenceDQNNetwork, self).__init__()
        
        # Input channels is the last dimension in reference format
        in_channels = input_shape[2]
        
        # Convolutional layers (same as reference)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        
        # Calculate conv output size
        self.conv_output_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        if initialize_weights:
            self._initialize_weights()
    
    def _get_conv_out_size(self, shape):
        """Calculate convolutional output size."""
        # Create dummy input in (H, W, C) format, convert to (C, H, W)
        dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
        dummy_output = self._forward_conv(dummy_input)
        return int(dummy_output.numel())
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.reshape(x.size(0), -1)
    
    def forward(self, x):
        """
        Forward pass compatible with reference implementation.
        
        Args:
            x: Input tensor of shape (batch_size, H, W, C) or (batch_size, C, H, W)
        
        Returns:
            Q-values for each action
        """
        # Handle different input formats
        if len(x.shape) == 4:
            if x.shape[1] == 84 and x.shape[3] in [4, 1]:  # (B, H, W, C) - reference format
                x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            elif x.shape[1] in [1, 4] and x.shape[2] == 84:  # (B, C, H, W) - PyTorch format
                pass  # Already correct
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        elif len(x.shape) == 3:  # Single sample (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
        
        # Normalize to [0, 1] if needed (reference does this)
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        # Convolutional layers
        x = self._forward_conv(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights like the reference implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class ReferenceCompatibleAgent:
    """
    DQN Agent that's fully compatible with the reference implementation.
    """
    
    def __init__(
        self,
        env,
        buffer_size=300000,
        batch_size=32,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        learning_rate=0.00025,
        target_update_interval=5000,
        device=None
    ):
        """Initialize reference-compatible agent."""
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_range = (epsilon_start, epsilon_end)
        self.epsilon = epsilon_start
        self.target_update_interval = target_update_interval
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.Q_network = ReferenceDQNNetwork(
            input_shape=(84, 84, 4),
            n_actions=env.action_space.n,
            initialize_weights=True
        ).to(self.device)
        
        self.Q_target_network = ReferenceDQNNetwork(
            input_shape=(84, 84, 4),
            n_actions=env.action_space.n,
            initialize_weights=False
        ).to(self.device)
        
        # Initialize target network
        self.update_target()
        self.Q_target_network.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.Q_network.parameters(), 
            lr=learning_rate, 
            eps=1.5e-4
        )
        
        # Loss function
        self.loss = nn.SmoothL1Loss()
        
        # Experience buffer (simplified)
        self.buffer = []
        self.buffer_size = buffer_size
        
        # Training statistics
        self.current_episode = 0
        self.step_count = 0
        self.action_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        self.loss_list = []
    
    def update_target(self):
        """Update target network."""
        self.Q_target_network.load_state_dict(self.Q_network.state_dict())
    
    def make_action(self, observation, test=False):
        """
        Select action using epsilon-greedy policy.
        Compatible with reference implementation interface.
        """
        if random.random() < (1 - self.epsilon) or test:
            # Greedy action
            with torch.no_grad():
                # Handle observation format
                if observation.shape == (84, 84, 4):  # Reference format
                    obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
                elif observation.shape == (4, 84, 84):  # PyTorch format
                    obs_tensor = torch.from_numpy(observation.transpose(1, 2, 0)).unsqueeze(0).to(self.device)
                else:
                    raise ValueError(f"Unexpected observation shape: {observation.shape}")
                
                action = self.Q_network(obs_tensor).argmax().item()
                self.action_counter[action] += 1
        else:
            # Random action
            action = self.env.action_space.sample()
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove oldest
        
        self.buffer.append(experience)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        import random
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.Q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.Q_target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * max_next_q_values * (~dones).unsqueeze(1))
        
        # Calculate loss
        loss = self.loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), 1.0)
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_list.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self, episode, total_episodes):
        """Update epsilon based on episode."""
        decay_episodes = total_episodes // 2
        if episode < decay_episodes:
            epsilon_step = (self.epsilon_range[0] - self.epsilon_range[1]) / decay_episodes
            self.epsilon = max(self.epsilon - epsilon_step, self.epsilon_range[1])


def create_reference_compatible_training():
    """Create a complete reference-compatible training setup."""
    
    # Import the reference wrapper
    import gymnasium as gym
    from atari_wrapper import wrap_deepmind
    
    # Create environment like reference
    env = gym.make('BreakoutNoFrameskip-v4', render_mode=None)
    env = wrap_deepmind(env, dim=84, clip_rewards=True, framestack=True, scale=False)
    
    # Create agent
    agent = ReferenceCompatibleAgent(
        env=env,
        buffer_size=300000,
        batch_size=32,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        learning_rate=0.00025,
        target_update_interval=5000
    )
    
    return agent, env
