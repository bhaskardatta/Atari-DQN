#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/examples/train_agent.py
"""
Example script for training the adaptive RL agent.
"""

import sys
import os
from pathlib import Path
import argparse
from dataclasses import asdict

# Register ALE environments
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.training.trainer import AdaptiveTrainer
from src.environment.breakout_env import DynamicBreakoutEnv
from src.dqn.agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description='Train adaptive RL agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger(
        name="training",
        level=config.logging.level,
        log_file=Path(config.paths.base_dir) / config.paths.logs_dir / 'training.log'
    )
    
    # Create trainer (it will create its own agent and environment)
    trainer = AdaptiveTrainer(
        config=config,
        experiment_name=f"training_{Path(args.config).stem}",
        save_dir=config.paths.base_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")
    
    # Start training
    print("Starting adaptive RL training...")
    print(f"Configuration: {args.config}")
    print(f"Device: {trainer.device}")
    print(f"Total episodes: {config.training.total_episodes}")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pth")
        print("Checkpoint saved as 'interrupted_checkpoint.pth'")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
