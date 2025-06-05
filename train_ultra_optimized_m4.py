#!/usr/bin/env python3
"""
Ultra-Optimized DQN Training for Apple Silicon M4
Maximum GPU utilization and training efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
import logging
import psutil
from collections import deque
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our optimizations
from m4_optimization_config import M4OptimizedConfig

# Set seeds for reproducibility
torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class UltraOptimizedDQN(nn.Module):
    """Ultra-optimized DQN Network for Apple Silicon M4"""
    
    def __init__(self, device, config):
        super(UltraOptimizedDQN, self).__init__()
        self.device = device
        
        # Get network configuration
        net_config = config['network_config']
        
        # Convolutional layers with optimized architecture
        self.conv1 = nn.Conv2d(4, net_config['conv_channels'][0], 
                              kernel_size=net_config['kernel_sizes'][0], 
                              stride=net_config['strides'][0], 
                              bias=net_config['use_bias'])
        self.conv2 = nn.Conv2d(net_config['conv_channels'][0], net_config['conv_channels'][1],
                              kernel_size=net_config['kernel_sizes'][1], 
                              stride=net_config['strides'][1], 
                              bias=net_config['use_bias'])
        self.conv3 = nn.Conv2d(net_config['conv_channels'][1], net_config['conv_channels'][2],
                              kernel_size=net_config['kernel_sizes'][2], 
                              stride=net_config['strides'][2], 
                              bias=net_config['use_bias'])
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, net_config['fc_hidden'])
        self.fc2 = nn.Linear(net_config['fc_hidden'], 4)  # 4 actions for Breakout
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm2d(net_config['conv_channels'][0])
        self.bn2 = nn.BatchNorm2d(net_config['conv_channels'][1])
        self.bn3 = nn.BatchNorm2d(net_config['conv_channels'][2])
        
        # Initialize weights
        self._initialize_weights()
        self.to(self.device)
        
        # Compile model for M4 if available
        if hasattr(torch, 'compile') and config.get('gpu_config', {}).get('compile_model', False):
            try:
                self = torch.compile(self, mode='max-autotune')
                print("✅ Model compiled for maximum performance")
            except:
                print("⚠️  Model compilation not available")
    
    def forward(self, x):
        """Optimized forward pass"""
        # Ensure proper device and dtype
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Normalize input
        if x.max() > 1.0:
            x = x / 255.0
        
        # Convolutional layers with batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten and fully connected
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class OptimizedReplayBuffer:
    """High-performance replay buffer optimized for M4"""
    
    def __init__(self, maxlen, batch_size, device):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.device = device
        self.buffer = deque(maxlen=maxlen)
        
    def push(self, experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self):
        """Sample batch with optimized tensor creation"""
        experiences = random.sample(self.buffer, self.batch_size)
        
        states = np.stack([e[0] for e in experiences])
        actions = [e[1] for e in experiences]
        rewards = [e[2] for e in experiences]
        next_states = np.stack([e[3] for e in experiences])
        dones = [e[4] for e in experiences]
        
        # Convert from (batch, H, W, C) to (batch, C, H, W) format for PyTorch
        if states.ndim == 4 and states.shape[-1] == 4:
            states = np.transpose(states, (0, 3, 1, 2)).copy()  # (B, H, W, C) -> (B, C, H, W)
        if next_states.ndim == 4 and next_states.shape[-1] == 4:
            next_states = np.transpose(next_states, (0, 3, 1, 2)).copy()  # (B, H, W, C) -> (B, C, H, W)
        
        # Create tensors directly on device
        states = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device, non_blocking=True)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device, non_blocking=True)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class M4OptimizedAgent:
    """Ultra-optimized DQN Agent for Apple Silicon M4"""
    
    def __init__(self, env, config_manager):
        self.env = env
        self.config = config_manager.training_config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🚀 Initializing M4 Optimized Agent on {self.device}")
        
        # Networks
        self.q_network = UltraOptimizedDQN(self.device, self.config)
        self.target_network = UltraOptimizedDQN(self.device, self.config)
        self.update_target_network()
        
        # Optimizer with M4 optimizations
        opt_config = config_manager.optimizer_config
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            eps=opt_config['eps'],
            betas=opt_config['betas']
        )
        
        # Loss function with better numerical stability
        self.loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        # Replay buffer
        self.replay_buffer = OptimizedReplayBuffer(
            self.config['training']['buffer_size'],
            self.config['training']['batch_size'],
            self.device
        )
        
        # Training parameters with numerical stability
        self.gamma = self.config['training']['gamma']
        self.epsilon = self.config['training']['epsilon_start']
        self.epsilon_end = self.config['training']['epsilon_end']
        self.epsilon_decay_steps = self.config['training']['epsilon_decay_steps']
        self.target_update_freq = self.config['training']['target_update_frequency']
        
        # Gradient clipping for stability
        self.max_grad_norm = 1.0
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.95)
        
        # Tracking
        self.step_count = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # Setup AMP if enabled
        self.use_amp = self.config['gpu_config']['use_amp']
        if self.use_amp and self.device.type == 'mps':
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            print("✅ Automatic Mixed Precision enabled")
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state, test=False):
        """Select action with epsilon-greedy policy"""
        if test or random.random() > self.epsilon:
            with torch.no_grad():
                # Convert from (H, W, C) to (C, H, W) format for PyTorch
                if state.ndim == 3 and state.shape[-1] == 4:
                    state = np.transpose(state, (2, 0, 1)).copy()  # (H, W, C) -> (C, H, W)
                
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, non_blocking=True)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, 3)
    
    def update_epsilon(self):
        """Update epsilon with linear decay"""
        if self.step_count < self.epsilon_decay_steps:
            decay_ratio = self.step_count / self.epsilon_decay_steps
            self.epsilon = self.config['training']['epsilon_start'] - \
                          (self.config['training']['epsilon_start'] - self.epsilon_end) * decay_ratio
        else:
            self.epsilon = self.epsilon_end
    
    def optimize_model(self):
        """Optimized model update with numerical stability"""
        if len(self.replay_buffer) < self.config['training']['batch_size']:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Clamp rewards for stability
        rewards = torch.clamp(rewards, -1.0, 1.0)
        
        # Forward pass with numerical stability checks
        try:
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0].detach()
                # Clamp Q-values to prevent explosion
                next_q = torch.clamp(next_q, -10.0, 10.0)
                target_q = rewards + (self.gamma * next_q * (1 - dones))
                target_q = torch.clamp(target_q, -10.0, 10.0)
            
            loss = self.loss_fn(current_q.squeeze(), target_q)
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print("⚠️  Loss is not finite, skipping update")
                return None
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            
            # Check gradients for NaN
            has_nan_grad = any(torch.isnan(p.grad).any() for p in self.q_network.parameters() if p.grad is not None)
            if has_nan_grad:
                print("⚠️  NaN gradients detected, skipping update")
                return None
            
            self.optimizer.step()
            self.scheduler.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"⚠️  Error in optimization: {e}")
            return None
    
    def run_episode(self, test=False):
        """Run a single episode"""
        state, _ = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = self.select_action(state, test)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            if not test:
                self.replay_buffer.push((state, action, reward, next_state, done))
                
                # Optimize model
                if self.step_count % self.config['training']['optimize_frequency'] == 0:
                    loss = self.optimize_model()
                
                # Update target network
                if self.step_count % self.target_update_freq == 0:
                    self.update_target_network()
                
                # Update epsilon
                self.update_epsilon()
                self.step_count += 1
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        return total_reward, steps
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting Ultra-Optimized M4 Training...")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        os.makedirs('checkpoints', exist_ok=True)
        
        # Fill replay buffer
        print("📦 Filling replay buffer...")
        while len(self.replay_buffer) < self.config['training']['batch_size'] * 10:
            state, _ = self.env.reset()
            done = False
            while not done and len(self.replay_buffer) < self.config['training']['batch_size'] * 10:
                action = random.randint(0, 3)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.push((state, action, reward, next_state, done))
                state = next_state
                if truncated:
                    break
        
        print(f"✅ Buffer filled with {len(self.replay_buffer)} experiences")
        
        start_time = time.time()
        episode_rewards = []
        
        # Training loop
        for episode in range(self.config['schedule']['num_episodes']):
            self.episode_count = episode
            
            # Run episode
            reward, steps = self.run_episode()
            episode_rewards.append(reward)
            
            # Logging
            if episode % self.config['schedule']['log_frequency'] == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                elapsed = time.time() - start_time
                eps_per_hour = episode / (elapsed / 3600) if elapsed > 0 else 0
                
                logger.info(f"Episode {episode:6d} | "
                           f"Reward: {reward:6.1f} | "
                           f"Avg: {avg_reward:6.1f} | "
                           f"ε: {self.epsilon:.3f} | "
                           f"Steps: {self.step_count:,} | "
                           f"Eps/hr: {eps_per_hour:.1f}")
            
            # Evaluation
            if episode % self.config['schedule']['eval_frequency'] == 0 and episode > 0:
                eval_rewards = []
                for _ in range(20):
                    eval_reward, _ = self.run_episode(test=True)
                    eval_rewards.append(eval_reward)
                
                avg_eval_reward = np.mean(eval_rewards)
                logger.info(f"🎯 Evaluation: {avg_eval_reward:.2f} (Episode {episode})")
                
                # Save best model
                if avg_eval_reward > self.best_reward:
                    self.best_reward = avg_eval_reward
                    torch.save(self.q_network.state_dict(), 'checkpoints/best_model_m4.pth')
                    logger.info(f"💾 New best model saved: {avg_eval_reward:.2f}")
            
            # Save checkpoint
            if episode % self.config['schedule']['save_frequency'] == 0 and episode > 0:
                torch.save(self.q_network.state_dict(), f'checkpoints/checkpoint_ep_{episode}.pth')
        
        # Save final model
        torch.save(self.q_network.state_dict(), 'checkpoints/final_model_m4.pth')
        total_time = time.time() - start_time
        logger.info(f"🏁 Training completed in {total_time/3600:.1f} hours!")

def main():
    """Main function"""
    # Initialize M4 configuration
    config_manager = M4OptimizedConfig()
    config_manager.validate_configuration()
    
    # Import environment
    import sys
    sys.path.append('.')
    from atari_wrapper import make_wrap_atari
    
    # Create environment
    env = make_wrap_atari('BreakoutNoFrameskip-v4', clip_rewards=True, render_mode=None)
    
    # Create and train agent
    agent = M4OptimizedAgent(env, config_manager)
    agent.train()

if __name__ == "__main__":
    main()
