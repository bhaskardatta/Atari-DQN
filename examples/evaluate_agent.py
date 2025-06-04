#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/examples/evaluate_agent.py
"""
Example script for evaluating the trained agent.
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger, get_experiment_logger
from src.environment.breakout_env import DynamicBreakoutEnv
from src.dqn.agent import DQNAgent
from src.analysis.visualization import VideoRecorder, save_visualization


def evaluate_agent(agent, env, num_episodes=10, record_video=False, video_path=None):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        record_video: Whether to record video
        video_path: Path to save video
        
    Returns:
        Dictionary with evaluation results
    """
    logger = get_experiment_logger("evaluation")
    
    agent.eval()  # Set to evaluation mode
    
    scores = []
    episode_lengths = []
    video_recorder = VideoRecorder() if record_video else None
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_score = 0
        episode_length = 0
        
        if record_video and episode == 0 and video_path:
            video_recorder.start_recording(
                video_path, 
                (env.observation_space.shape[1], env.observation_space.shape[0])
            )
        
        done = False
        while not done:
            # Select action (no exploration during evaluation)
            action = agent.select_action(state, explore=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Record frame if needed
            if record_video and episode == 0 and video_recorder.recording:
                # Convert state to RGB for recording
                if len(state.shape) == 3 and state.shape[0] == 1:
                    frame = np.repeat(state[0:1], 3, axis=0).transpose(1, 2, 0)
                elif len(state.shape) == 3:
                    frame = state.transpose(1, 2, 0)
                else:
                    frame = np.stack([state] * 3, axis=2)
                
                frame = (frame * 255).astype(np.uint8)
                video_recorder.add_frame(frame)
            
            episode_score += reward
            episode_length += 1
            state = next_state
        
        if record_video and episode == 0 and video_recorder.recording:
            video_recorder.stop_recording()
        
        scores.append(episode_score)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode + 1}: Score = {episode_score}, Length = {episode_length}")
    
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'scores': scores,
        'lengths': episode_lengths
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--record-video', action='store_true',
                        help='Record video of first episode')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logger(name="evaluation", level=config.logging.level)
    logger = get_experiment_logger("evaluation")
    
    # Create environment
    env = DynamicBreakoutEnv(
        env_name=config.environment.env_name,
        frame_stack=config.environment.frame_stack,
        frame_skip=config.environment.frame_skip,
        frame_size=config.environment.frame_size,
        grayscale=config.environment.grayscale,
        difficulty_config=config.difficulty
    )
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n,
        config=config.agent,
        network_config=config.network,
        replay_config=config.replay_buffer,
        device=args.device
    )
    
    # Load trained model
    logger.info(f"Loading model from {args.checkpoint}")
    agent.load_checkpoint(args.checkpoint)
    
    # Set up video recording
    video_path = None
    if args.record_video:
        video_dir = Path(config.paths.base_dir) / config.paths.videos_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / 'evaluation.mp4')
    
    # Evaluate agent
    logger.info(f"Evaluating agent for {args.episodes} episodes")
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        record_video=args.record_video,
        video_path=video_path
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {args.episodes}")
    print(f"Mean Score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"Score Range: [{results['min_score']:.2f}, {results['max_score']:.2f}]")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    
    if args.record_video and video_path:
        print(f"Video saved to: {video_path}")
    
    print("="*50)
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()
