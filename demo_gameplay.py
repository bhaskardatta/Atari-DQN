#!/usr/bin/env python3
"""
Demo script showing the adaptive RL agent playing Atari Breakout.
This script loads a trained model and demonstrates gameplay with visual rendering.
"""

import argparse
import numpy as np
import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.dqn.agent import DQNAgent
from src.environment.breakout_env import DynamicBreakoutEnv
from src.utils.config import load_config


def create_demo_agent():
    """Create a demo agent with random policy for demonstration."""
    # Use a simple configuration for demo
    demo_config = {
        'state_shape': (4, 84, 84),
        'n_actions': 4,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 0.0,  # No exploration for demo
        'epsilon_end': 0.0,
        'epsilon_decay': 1.0,
        'target_update_freq': 1000,
        'batch_size': 32,
        'buffer_size': 10000,
        'device': 'cpu'
    }
    
    agent = DQNAgent(**demo_config)
    return agent


def load_trained_agent(checkpoint_path: str):
    """Load a trained agent from checkpoint."""
    try:
        # Load the model state
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create agent with appropriate configuration
        agent_config = {
            'state_shape': (4, 84, 84),
            'n_actions': 4,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 0.0,  # No exploration for demo
            'epsilon_end': 0.0,
            'epsilon_decay': 1.0,
            'target_update_freq': 1000,
            'batch_size': 32,
            'buffer_size': 10000,
            'device': 'cpu'
        }
        
        agent = DQNAgent(**agent_config)
        
        # Load the trained weights - handle different checkpoint formats
        if 'q_network_state_dict' in checkpoint:
            # Our format from training
            agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        elif 'model_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the model state dict
            agent.q_network.load_state_dict(checkpoint)
        
        agent.q_network.eval()
        print(f"✅ Loaded trained agent from {checkpoint_path}")
        return agent
        
    except Exception as e:
        print(f"⚠️  Could not load trained model ({e})")
        print("🎮 Using random agent for demonstration instead")
        return create_demo_agent()


def run_demo(agent, episodes=3, render=True, max_steps=1000):
    """Run the demo gameplay."""
    
    # Create environment with rendering enabled
    env = DynamicBreakoutEnv(
        render_mode='human' if render else 'rgb_array',
        difficulty_config={'difficulty_factor': 0.5}  # Moderate difficulty for demo
    )
    
    print(f"\n🎮 Starting Breakout Demo")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Rendering: {'ON' if render else 'OFF'}")
    print(f"Dynamic difficulty: ON")
    print("-" * 50)
    
    total_reward = 0
    total_steps = 0
    
    for episode in range(episodes):
        print(f"\n🏁 Episode {episode + 1}")
        
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Select action using the agent
            action = agent.select_action(state, training=False)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update statistics
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Print progress every 100 steps
            if step % 100 == 0 and step > 0:
                print(f"  Step {step}: Reward = {episode_reward:.1f}")
            
            # Update state
            state = next_state
            
            # Add small delay for human viewing
            if render:
                time.sleep(0.02)  # 50 FPS
            
            if done:
                break
        
        total_reward += episode_reward
        
        print(f"  ✅ Episode {episode + 1} complete!")
        print(f"     Steps: {episode_steps}")
        print(f"     Reward: {episode_reward:.2f}")
        
        # Short pause between episodes
        if render and episode < episodes - 1:
            print("  ⏸️  Pausing 2 seconds before next episode...")
            time.sleep(2)
    
    env.close()
    
    # Final statistics
    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    
    print("\n" + "=" * 50)
    print("🏆 DEMO COMPLETE!")
    print(f"Total Episodes: {episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print("=" * 50)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Demo: Watch AI play Breakout')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to play')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visual rendering (faster)')
    parser.add_argument('--config', type=str, default='configs/quick_test.yaml',
                       help='Configuration file to use')
    
    args = parser.parse_args()
    
    print("🚀 Adaptive RL Breakout Demo")
    print("=" * 50)
    
    # Try to load a trained model if specified
    if args.model:
        agent = load_trained_agent(args.model)
    else:
        # Try to find a trained model in the results
        possible_models = [
            'test_results/final_model.pth',
            'test_results/best_model.pth',
            'results/models/dqn_final.pth',
            'results/models/best_model.pth'
        ]
        
        model_found = False
        for model_path in possible_models:
            if Path(model_path).exists():
                print(f"🔍 Found trained model: {model_path}")
                agent = load_trained_agent(model_path)
                model_found = True
                break
        
        if not model_found:
            print("🎯 No trained model found, using random agent for demo")
            agent = create_demo_agent()
    
    # Run the demo
    try:
        run_demo(
            agent=agent,
            episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
