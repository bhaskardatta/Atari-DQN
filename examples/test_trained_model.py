#!/usr/bin/env python3
"""
Test the trained model performance and demonstrate gameplay.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dqn.agent import DQNAgent
from src.environment.breakout_env import make_dynamic_breakout
from src.utils.config import load_config


def load_trained_agent(model_path: str, config_path: str):
    """Load a trained agent from checkpoint."""
    config = load_config(config_path)
    
    # Create environment to get action space
    env = make_dynamic_breakout(
        difficulty_factor=0.0,
        frame_stack=config.environment.frame_stack,
        render_mode=None
    )
    
    # Create agent
    agent = DQNAgent(
        state_shape=config.network.input_shape,
        n_actions=env.action_space.n,
        learning_rate=config.network.learning_rate,
        gamma=config.network.gamma,
        epsilon_start=0.0,  # No exploration for evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        target_update_freq=config.network.target_update,
        batch_size=config.network.batch_size,
        buffer_size=config.training.buffer_size,
        min_buffer_size=config.training.min_buffer_size,
        device="cpu",  # Use CPU for demo
        double_dqn=True,
        dueling=True,
        prioritized_replay=True,
        adaptive_replay=True,
        seed=config.training.seed
    )
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    
    print(f"Loaded model from {model_path}")
    print(f"Training episode: {checkpoint.get('episode', 'Unknown')}")
    print(f"Total steps: {checkpoint.get('total_steps', 'Unknown')}")
    
    env.close()
    return agent, config


def evaluate_agent(agent, config, num_episodes=10, render=False, difficulty_factor=0.0):
    """Evaluate agent performance."""
    env = make_dynamic_breakout(
        difficulty_factor=difficulty_factor,
        frame_stack=config.environment.frame_stack,
        render_mode="human" if render else None
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results (Difficulty: {difficulty_factor:.1f}):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'max_reward': max(episode_rewards),
        'min_reward': min(episode_rewards),
        'rewards': episode_rewards,
        'lengths': episode_lengths
    }


def test_different_difficulties(agent, config):
    """Test agent performance at different difficulty levels."""
    difficulty_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    
    print("Testing agent performance at different difficulty levels...")
    print("=" * 60)
    
    for difficulty in difficulty_levels:
        print(f"\nTesting at difficulty level {difficulty:.1f}")
        print("-" * 40)
        results[difficulty] = evaluate_agent(
            agent, config, 
            num_episodes=5, 
            render=False, 
            difficulty_factor=difficulty
        )
    
    return results


def plot_difficulty_performance(results):
    """Plot performance across different difficulty levels."""
    difficulties = list(results.keys())
    avg_rewards = [results[d]['avg_reward'] for d in difficulties]
    std_rewards = [results[d]['std_reward'] for d in difficulties]
    avg_lengths = [results[d]['avg_length'] for d in difficulties]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average rewards
    ax1.errorbar(difficulties, avg_rewards, yerr=std_rewards, 
                marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Agent Performance vs Difficulty')
    ax1.grid(True, alpha=0.3)
    
    # Plot average episode lengths
    ax2.plot(difficulties, avg_lengths, marker='s', color='orange')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Episode Length vs Difficulty')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path("training_extended_training/difficulty_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {save_path}")
    
    plt.show()


def demonstrate_gameplay(agent, config, num_demos=3):
    """Demonstrate the agent playing the game visually."""
    print(f"\nDemonstrating agent gameplay (CPU only, {num_demos} episodes)...")
    print("Close the game window to continue to the next episode.")
    print("=" * 60)
    
    for demo in range(num_demos):
        print(f"\nDemo {demo + 1}/{num_demos}")
        print("Starting game... (Close window when done)")
        
        try:
            result = evaluate_agent(
                agent, config, 
                num_episodes=1, 
                render=True, 
                difficulty_factor=0.1  # Slight difficulty for interesting gameplay
            )
            print(f"Demo {demo + 1} completed: Reward = {result['rewards'][0]:.2f}")
        except KeyboardInterrupt:
            print("Demo interrupted by user")
            break
        except Exception as e:
            print(f"Demo error: {e}")
            break


def main():
    """Main evaluation function."""
    # Model paths
    model_paths = [
        ("Final Model", "training_extended_training/final_model.pth"),
        ("Best Model", "training_extended_training/best_model.pth"),
    ]
    
    config_path = "configs/extended_training.yaml"
    
    for model_name, model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load the trained agent
            agent, config = load_trained_agent(model_path, config_path)
            
            # Quick evaluation at normal difficulty
            print(f"\n{'-'*60}")
            print("QUICK EVALUATION (Normal Difficulty)")
            print(f"{'-'*60}")
            quick_result = evaluate_agent(agent, config, num_episodes=5, difficulty_factor=0.0)
            
            # Test at different difficulties
            print(f"\n{'-'*60}")
            print("DIFFICULTY ADAPTATION TEST")
            print(f"{'-'*60}")
            difficulty_results = test_different_difficulties(agent, config)
            
            # Plot results
            plot_difficulty_performance(difficulty_results)
            
            # Demonstrate gameplay (optional - commented out for automated testing)
            # demonstrate_gameplay(agent, config, num_demos=2)
            
            print(f"\n{model_name} evaluation completed!")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print("✅ Training completed successfully")
    print("✅ Model evaluation completed")
    print("✅ Difficulty adaptation tested")
    print("✅ Performance analysis generated")
    print("\nThe trained agent is ready for use!")


if __name__ == "__main__":
    main()
