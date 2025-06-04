#!/usr/bin/env python3
"""
Simplified and effective DQN training script.
Combines the best practices from the reference implementation with our existing infrastructure.
"""

import sys
import os
import time
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import reference wrapper and our existing components
from atari_wrapper import make_wrap_atari
from reference_dqn import ReferenceDQNNetwork


class SimpleExperienceBuffer:
    """Simple experience replay buffer."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch from buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class EffectiveDQNAgent:
    """
    Simplified but effective DQN agent based on reference implementation.
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
        """Initialize effective DQN agent."""
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.target_update_interval = target_update_interval
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Networks - using reference-compatible network
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
        
        # Optimizer (using reference parameters)
        self.optimizer = torch.optim.Adam(
            self.Q_network.parameters(), 
            lr=learning_rate, 
            eps=1.5e-4
        )
        
        # Loss function
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # Experience buffer
        self.buffer = SimpleExperienceBuffer(buffer_size)
        
        # Training statistics
        self.step_count = 0
        self.action_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        self.loss_list = []
        
        print(f"Agent initialized with {sum(p.numel() for p in self.Q_network.parameters())} parameters")
    
    def update_target(self):
        """Update target network."""
        self.Q_target_network.load_state_dict(self.Q_network.state_dict())
        print("🎯 Target network updated")
    
    def select_action(self, observation, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Random exploration
            action = self.env.action_space.sample()
        else:
            # Greedy action selection
            with torch.no_grad():
                # Convert observation to tensor and add batch dimension
                obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
                q_values = self.Q_network(obs_tensor)
                action = q_values.argmax().item()
        
        # Track action distribution
        self.action_counter[action] += 1
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
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
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), 1.0)
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_list.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self, episode, total_episodes):
        """Update epsilon with linear decay."""
        decay_episodes = total_episodes // 2
        if episode < decay_episodes:
            epsilon_step = (self.epsilon_start - self.epsilon_end) / decay_episodes
            self.epsilon = max(self.epsilon - epsilon_step, self.epsilon_end)


def train_effective_agent():
    """Train an effective DQN agent using simplified but proven methods."""
    
    print("🚀 EFFECTIVE DQN TRAINING")
    print("=" * 60)
    
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Training configuration (based on reference implementation)
    config = {
        'episodes': 3000,
        'max_steps': 27000,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'buffer_size': 300000,
        'batch_size': 32,
        'target_update_freq': 5000,
        'optimize_interval': 4,  # Train every 4 steps like reference
        'eval_freq': 250,
        'save_freq': 500,
        'min_buffer_size': 50000
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create environment using reference wrapper
    print("\\nCreating environment...")
    env = make_wrap_atari('BreakoutNoFrameskip-v4', clip_rewards=True, render_mode=None)
    print(f"Environment created: {env.observation_space.shape} -> {env.action_space.n} actions")
    
    # Create agent
    print("\\nCreating agent...")
    agent = EffectiveDQNAgent(
        env=env,
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        learning_rate=config['learning_rate'],
        target_update_interval=config['target_update_freq']
    )
    
    # Create save directory
    save_dir = Path("effective_training")
    save_dir.mkdir(exist_ok=True)
    
    print("\\n🎯 Starting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    # Pre-fill buffer with random experiences
    print("Filling replay buffer...")
    fill_steps = 0
    while len(agent.buffer) < config['min_buffer_size']:
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000 and fill_steps < 100000:
            action = env.action_space.sample()  # Random action
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.store_experience(obs, action, reward, next_obs, done or truncated)
            obs = next_obs
            fill_steps += 1
            steps += 1
            
            if fill_steps % 10000 == 0:
                print(f"  Buffer: {len(agent.buffer)}/{config['min_buffer_size']}")
    
    print(f"✅ Buffer filled with {len(agent.buffer)} experiences")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    recent_losses = []
    
    # Main training loop
    for episode in range(config['episodes']):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        # Reset episode stats
        agent.action_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for step in range(config['max_steps']):
            # Select and execute action
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done or truncated)
            
            # Train every few steps
            if step % config['optimize_interval'] == 0:
                loss = agent.train_step()
                if loss > 0:
                    episode_losses.append(loss)
            
            # Update target network
            if agent.step_count % agent.target_update_interval == 0:
                agent.update_target()
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            agent.step_count += 1
            
            if done or truncated:
                break
        
        # Update epsilon
        agent.update_epsilon(episode, config['episodes'])
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        recent_losses.append(avg_loss)
        
        # Print progress
        if episode % 50 == 0 or episode < 10:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_length_100 = np.mean(episode_lengths[-100:])
            avg_loss_100 = np.mean(recent_losses[-100:])
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg-100: {avg_reward_100:6.1f} | "
                  f"Length: {episode_length:5d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
            print(f"  Actions: {agent.action_counter}")
        
        # Evaluation
        if episode % config['eval_freq'] == 0 and episode > 0:
            eval_reward = evaluate_agent(agent, env, 10)
            print(f"\\n📊 Evaluation (Episode {episode}): {eval_reward:.1f} avg reward")
            
            if eval_reward > 100:  # Decent performance threshold
                print("🎯 Good performance detected!")
        
        # Save checkpoint
        if episode % config['save_freq'] == 0 and episode > 0:
            save_path = save_dir / f"checkpoint_episode_{episode}.pth"
            torch.save({
                'episode': episode,
                'q_network_state_dict': agent.Q_network.state_dict(),
                'target_network_state_dict': agent.Q_target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'step_count': agent.step_count,
                'episode_rewards': episode_rewards,
                'config': config
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")
    
    # Save final model
    final_path = save_dir / "final_effective_model.pth"
    torch.save({
        'episode': config['episodes'],
        'q_network_state_dict': agent.Q_network.state_dict(),
        'target_network_state_dict': agent.Q_target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'step_count': agent.step_count,
        'episode_rewards': episode_rewards,
        'config': config
    }, final_path)
    
    # Final evaluation
    print("\\n🎊 Training completed!")
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    final_eval = evaluate_agent(agent, env, 100)
    print(f"\\n🏆 Final evaluation (100 episodes): {final_eval:.1f} avg reward")
    
    # Performance summary
    print("\\n📈 Training Summary:")
    print(f"  Episodes: {len(episode_rewards)}")
    print(f"  Final reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"  Best reward: {max(episode_rewards):.1f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Total steps: {agent.step_count}")
    
    env.close()
    return agent, final_path


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance without exploration."""
    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 27000:
            action = agent.select_action(obs, training=False)  # No exploration
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


if __name__ == "__main__":
    agent, model_path = train_effective_agent()
    print(f"\\n✅ Training complete! Final model saved to: {model_path}")
    print("\\n🎮 You can now test this model with a demo script.")
