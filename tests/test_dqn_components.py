"""
Unit tests for DQN components.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.dqn.network import DQNNetwork, NoisyDQNNetwork
from src.dqn.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, AdaptiveReplayBuffer
from src.dqn.agent import DQNAgent


class TestDQNNetwork(unittest.TestCase):
    """Test DQN network implementations."""
    
    def setUp(self):
        self.state_dim = (4, 84, 84)
        self.action_dim = 4
        self.config = {
            'hidden_dims': [512, 512],
            'activation': 'relu',
            'dropout': 0.0
        }
    
    def test_dqn_network_creation(self):
        """Test DQN network creation and forward pass."""
        network = DQNNetwork(self.state_dim, self.action_dim, dueling=False)
        
        # Test forward pass
        batch_size = 2
        state = torch.randn(batch_size, *self.state_dim)
        output = network(state)
        
        self.assertEqual(output.shape, (batch_size, self.action_dim))
    
    def test_dueling_dqn_creation(self):
        """Test Dueling DQN network creation and forward pass."""
        network = DQNNetwork(self.state_dim, self.action_dim, dueling=True)
        
        # Test forward pass
        batch_size = 2
        state = torch.randn(batch_size, *self.state_dim)
        output = network(state)
        
        self.assertEqual(output.shape, (batch_size, self.action_dim))
    
    def test_noisy_dqn_creation(self):
        """Test Noisy DQN network creation and forward pass."""
        network = NoisyDQNNetwork(self.state_dim, self.action_dim, dueling=True)
        
        # Test forward pass
        batch_size = 2
        state = torch.randn(batch_size, *self.state_dim)
        output = network(state)
        
        self.assertEqual(output.shape, (batch_size, self.action_dim))
    
    def test_network_parameter_count(self):
        """Test that networks have reasonable parameter counts."""
        networks = [
            DQNNetwork(self.state_dim, self.action_dim, dueling=False),
            DQNNetwork(self.state_dim, self.action_dim, dueling=True),
            NoisyDQNNetwork(self.state_dim, self.action_dim, dueling=True)
        ]
        
        for network in networks:
            param_count = sum(p.numel() for p in network.parameters())
            self.assertGreater(param_count, 1000)  # Should have reasonable number of parameters


class TestReplayBuffer(unittest.TestCase):
    """Test replay buffer implementations."""
    
    def setUp(self):
        self.capacity = 1000
        self.state_dim = (4, 84, 84)
        self.batch_size = 32
    
    def test_basic_replay_buffer(self):
        """Test basic replay buffer functionality."""
        buffer = ReplayBuffer(self.capacity)
        
        # Test empty buffer
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.is_ready(self.batch_size))
        
        # Add experiences
        for i in range(100):
            state = np.random.randn(*self.state_dim)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_dim)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 100)
        self.assertTrue(buffer.is_ready(self.batch_size))
        
        # Test sampling
        batch = buffer.sample(self.batch_size)
        self.assertEqual(len(batch), 5)  # state, action, reward, next_state, done
        self.assertEqual(batch[0].shape[0], self.batch_size)
    
    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer functionality."""
        buffer = PrioritizedReplayBuffer(self.capacity, alpha=0.6, beta=0.4)
        
        # Add experiences
        for i in range(100):
            state = np.random.randn(*self.state_dim)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_dim)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        # Test sampling with priorities
        batch = buffer.sample(self.batch_size)
        self.assertEqual(len(batch), 7)  # includes weights and indices
        
        # Test priority updates
        indices = batch[5]
        priorities = np.abs(np.random.randn(len(indices))) + 0.1
        buffer.update_priorities(indices, priorities)
    
    def test_adaptive_replay_buffer(self):
        """Test adaptive replay buffer functionality."""
        buffer = AdaptiveReplayBuffer(self.capacity, num_conditions=3)
        
        # Add experiences with different conditions
        for condition in range(3):
            for i in range(50):
                state = np.random.randn(*self.state_dim)
                action = np.random.randint(0, 4)
                reward = np.random.randn()
                next_state = np.random.randn(*self.state_dim)
                done = np.random.choice([True, False])
                
                buffer.push(state, action, reward, next_state, done, condition)
        
        # Test condition-aware sampling
        batch = buffer.sample_balanced(self.batch_size)
        self.assertEqual(len(batch), 5)


class TestDQNAgent(unittest.TestCase):
    """Test DQN agent functionality."""
    
    def setUp(self):
        self.state_dim = (4, 84, 84)
        self.action_dim = 4
    
    def test_agent_creation(self):
        """Test DQN agent creation."""
        agent = DQNAgent(
            state_shape=self.state_dim,
            n_actions=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=32,
            buffer_size=1000,
            device='cpu'
        )
        
        self.assertEqual(agent.n_actions, self.action_dim)
        self.assertIsNotNone(agent.q_network)
        self.assertIsNotNone(agent.target_network)
        self.assertIsNotNone(agent.replay_buffer)
    
    def test_action_selection(self):
        """Test action selection functionality."""
        agent = DQNAgent(
            state_shape=self.state_dim,
            n_actions=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=32,
            buffer_size=1000,
            device='cpu'
        )
        
        state = np.random.randn(*self.state_dim)
        
        # Test exploratory action selection
        action = agent.select_action(state, training=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        
        # Test greedy action selection
        action = agent.select_action(state, training=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
    
    def test_experience_storage(self):
        """Test experience storage and replay."""
        agent = DQNAgent(
            state_shape=self.state_dim,
            n_actions=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=32,
            buffer_size=1000,
            device='cpu'
        )
        
        # Store experiences
        for i in range(100):
            state = np.random.randn(*self.state_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_dim)
            done = np.random.choice([True, False])
            
            agent.store_experience(state, action, reward, next_state, done)
        
        self.assertEqual(len(agent.replay_buffer), 100)
    
    def test_training_step(self):
        """Test training step functionality."""
        agent = DQNAgent(
            state_shape=self.state_dim,
            n_actions=self.action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=32,
            buffer_size=1000,
            min_buffer_size=100,  # Smaller minimum for testing
            device='cpu'
        )
        
        # Fill replay buffer with enough samples
        for i in range(150):  # More than min_buffer_size
            state = np.random.randn(*self.state_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_dim)
            done = np.random.choice([True, False])
            
            agent.store_experience(state, action, reward, next_state, done)
        
        # Test training step
        # Get a baseline loss
        baseline_losses = []
        for _ in range(3):
            loss = agent.train_step()
            if loss is not None:
                baseline_losses.append(loss)
        
        # Training should produce some loss values
        self.assertTrue(len(baseline_losses) > 0)
        self.assertTrue(all(isinstance(loss, (int, float)) for loss in baseline_losses))


if __name__ == '__main__':
    unittest.main()
