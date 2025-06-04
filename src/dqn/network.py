"""
Neural network architecture for the DQN agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Atari Breakout.
    
    Architecture:
    - 3 Convolutional layers for feature extraction
    - 2 Fully connected layers for Q-value estimation
    - Optional dueling architecture
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        n_actions: int = 4,
        hidden_dims: list = [512, 512],
        dueling: bool = True
    ):
        """
        Initialize the DQN network.
        
        Args:
            input_shape: Shape of input observations (channels, height, width)
            n_actions: Number of possible actions
            hidden_dims: Hidden layer dimensions
            dueling: Whether to use dueling architecture
        """
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.dueling = dueling
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers
        if self.dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], n_actions)
            )
        else:
            # Standard DQN architecture
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], n_actions)
            )
    
    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculate the output size of convolutional layers."""
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return int(dummy_output.numel())
    
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, *input_shape)
        
        Returns:
            Q-values for each action
        """
        # Normalize input to [0, 1] if needed
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        # Convolutional layers
        x = self._forward_conv(x)
        
        if self.dueling:
            # Dueling architecture
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Combine value and advantage streams
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_values = self.fc(x)
        
        return q_values


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration in DQN.
    Implements factorized Gaussian noise as described in "Noisy Networks for Exploration".
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize noisy linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize learnable parameters."""
        mu_range = 1 / (self.in_features ** 0.5)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))
    
    def reset_noise(self):
        """Reset noise buffers."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights and biases."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class NoisyDQNNetwork(DQNNetwork):
    """
    DQN Network with noisy layers for exploration.
    Replaces the final linear layers with noisy variants.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (4, 84, 84),
        n_actions: int = 4,
        hidden_dims: list = [512, 512],
        dueling: bool = True,
        std_init: float = 0.5
    ):
        """Initialize noisy DQN network."""
        # Initialize parent class but replace final layers
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.dueling = dueling
        
        # Convolutional layers (same as standard DQN)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Noisy fully connected layers
        if self.dueling:
            # Dueling architecture with noisy layers
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                NoisyLinear(hidden_dims[1], 1, std_init)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                NoisyLinear(hidden_dims[1], n_actions, std_init)
            )
        else:
            # Standard architecture with noisy final layer
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                NoisyLinear(hidden_dims[1], n_actions, std_init)
            )
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
