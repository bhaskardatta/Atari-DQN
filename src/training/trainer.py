"""
Main training loop for the adaptive DQN agent.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

from ..utils.config import Config, load_config
from ..utils.logging import get_experiment_logger, MetricsLogger, TensorBoardLogger
from ..environment.breakout_env import make_dynamic_breakout
from ..environment.difficulty import CurriculumDifficulty
from ..dqn.agent import DQNAgent
from .adaptation import AdaptationDetector
from ..analysis.performance import PerformanceAnalyzer


class AdaptiveTrainer:
    """
    Main trainer for the adaptive DQN agent.
    Handles training loop, curriculum learning, and adaptation detection.
    """
    
    def __init__(
        self,
        config: Config,
        experiment_name: str = "adaptive_breakout",
        save_dir: str = "results"
    ):
        """
        Initialize adaptive trainer.
        
        Args:
            config: Training configuration
            experiment_name: Name of the experiment
            save_dir: Directory to save results
        """
        self.config = config
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device(
            config.training.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize logging
        self.logger = get_experiment_logger(experiment_name)
        self.metrics_logger = MetricsLogger(str(self.save_dir), experiment_name)
        self.tensorboard_logger = TensorBoardLogger(str(self.save_dir), experiment_name)
        
        # Initialize environment
        self.env = make_dynamic_breakout(
            difficulty_factor=0.0,  # Start with normal difficulty
            frame_stack=config.environment.frame_stack,
            render_mode=config.environment.render_mode
        )
        
        # Initialize agent
        self.agent = DQNAgent(
            state_shape=config.network.input_shape,
            n_actions=self.env.action_space.n,
            learning_rate=config.network.learning_rate,
            gamma=config.network.gamma,
            epsilon_start=config.network.eps_start,
            epsilon_end=config.network.eps_end,
            epsilon_decay=config.network.eps_decay,
            target_update_freq=config.network.target_update,
            batch_size=config.network.batch_size,
            buffer_size=config.training.buffer_size,
            min_buffer_size=config.training.min_buffer_size,
            device=str(self.device),
            double_dqn=True,
            dueling=True,
            prioritized_replay=True,
            adaptive_replay=True,
            seed=config.training.seed
        )
        
        # Initialize curriculum learning
        if config.curriculum.enable:
            # Convert stages to expected format and calculate stage duration
            curriculum_stages = []
            for stage in config.curriculum.stages:
                curriculum_stages.append({
                    "name": stage["name"],
                    "difficulty_factor": stage.get("difficulty_multiplier", 1.0) - 1.0  # Convert to 0-1 scale
                })
            
            # Calculate stage duration from episodes (assuming fixed episodes per stage)
            avg_episodes_per_stage = int(np.mean([stage.get("episodes", 100) for stage in config.curriculum.stages]))
            
            self.curriculum = CurriculumDifficulty(
                stages=curriculum_stages,
                stage_duration=avg_episodes_per_stage,
                performance_threshold=config.curriculum.advancement_threshold,
                patience=5  # Default patience value
            )
        else:
            self.curriculum = None
        
        # Initialize adaptation detector
        self.adaptation_detector = AdaptationDetector(
            window_size=100,
            sensitivity=0.2,
            min_episodes=50
        )
        
        # Initialize performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_avg_reward = float('-inf')
        self.training_start_time = time.time()
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_scores = []
        
        self.logger.info(f"Initialized trainer for experiment: {experiment_name}")
        self.logger.info(f"Configuration: {config.to_dict()}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        try:
            while self.episode < self.config.training.total_episodes:
                # Train one episode
                episode_reward, episode_length, episode_loss = self._train_episode()
                
                # Update curriculum
                current_difficulty = self.curriculum.update_episode(episode_reward)
                self.env.set_difficulty_factor(current_difficulty)
                
                # Detect adaptation events
                adaptation_info = self.adaptation_detector.update(
                    episode_reward, current_difficulty
                )
                
                # Handle adaptation if needed
                if adaptation_info['change_detected']:
                    self.logger.info(f"Environmental change detected at episode {self.episode}")
                    self.agent.adapt_to_change()
                    
                    # Log adaptation event
                    self.metrics_logger.log_adaptation_event(
                        episode=self.episode,
                        step=self.total_steps,
                        change_detected=True,
                        change_type=adaptation_info['change_type'],
                        adaptation_score=adaptation_info['adaptation_score'],
                        recovery_time=adaptation_info.get('recovery_time', 0)
                    )
                
                # Evaluation
                if self.episode % self.config.training.eval_frequency == 0:
                    eval_reward = self._evaluate()
                    self.evaluation_scores.append(eval_reward)
                    
                    # Log performance evaluation
                    self.metrics_logger.log_performance_evaluation(
                        episode=self.episode,
                        avg_reward=eval_reward,
                        std_reward=np.std(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0.0,
                        success_rate=self._calculate_success_rate(),
                        difficulty_level=current_difficulty
                    )
                    
                    # Check if best performance
                    if eval_reward > self.best_avg_reward:
                        self.best_avg_reward = eval_reward
                        self._save_checkpoint('best_model.pth')
                
                # Save checkpoint
                if self.episode % self.config.training.save_frequency == 0:
                    self._save_checkpoint(f'checkpoint_episode_{self.episode}.pth')
                
                # Log training metrics
                self._log_training_metrics(episode_reward, episode_length, episode_loss)
                
                # Print progress
                if self.episode % 100 == 0:
                    self._print_progress()
                
                self.episode += 1
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self._finalize_training()
    
    def _train_episode(self) -> tuple:
        """
        Train one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length, average_loss)
        """
        state, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_losses = []
        
        while episode_length < self.config.training.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Get current difficulty condition for adaptive replay
            difficulty_level = info.get('difficulty', {}).get('difficulty_factor', 0.0)
            condition = min(int(difficulty_level * 4), 3)  # Ensure condition is 0-3
            
            # Store experience and train
            loss = self.agent.step(state, action, reward, next_state, done, condition)
            
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                break
        
        # Record episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Calculate average loss
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        return episode_reward, episode_length, avg_loss
    
    def _evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate agent performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Average evaluation reward
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while episode_length < self.config.training.max_steps_per_episode:
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        self.logger.info(f"Evaluation (Episode {self.episode}): {avg_reward:.2f} ± {np.std(eval_rewards):.2f}")
        
        return avg_reward
    
    def _calculate_success_rate(self, threshold: float = 10.0) -> float:
        """Calculate success rate based on recent episodes."""
        if len(self.episode_rewards) < 10:
            return 0.0
        
        recent_rewards = self.episode_rewards[-100:]
        successful_episodes = sum(1 for r in recent_rewards if r >= threshold)
        return successful_episodes / len(recent_rewards)
    
    def _log_training_metrics(self, episode_reward: float, episode_length: int, episode_loss: float):
        """Log training metrics to various loggers."""
        # Agent statistics
        agent_stats = self.agent.get_stats()
        
        # Log to metrics logger
        self.metrics_logger.log_training_step(
            episode=self.episode,
            step=self.total_steps,
            reward=episode_reward,
            epsilon=agent_stats['epsilon'],
            loss=episode_loss,
            q_value=0.0,  # Would need to track this separately
            duration=episode_length
        )
        
        # Log to TensorBoard
        self.tensorboard_logger.log_scalar('reward/episode', episode_reward, self.episode)
        self.tensorboard_logger.log_scalar('episode/length', episode_length, self.episode)
        self.tensorboard_logger.log_scalar('loss/episode', episode_loss, self.episode)
        self.tensorboard_logger.log_scalar('agent/epsilon', agent_stats['epsilon'], self.episode)
        self.tensorboard_logger.log_scalar('agent/buffer_size', agent_stats['buffer_size'], self.episode)
        
        # Log curriculum information
        stage_info = self.curriculum.get_current_stage_info()
        self.tensorboard_logger.log_scalar('curriculum/stage', stage_info['stage_index'], self.episode)
        self.tensorboard_logger.log_scalar('curriculum/difficulty', stage_info['difficulty_factor'], self.episode)
        
        # Log environment statistics
        env_stats = self.env.get_difficulty_statistics()
        self.tensorboard_logger.log_scalar('environment/total_changes', env_stats['total_changes'], self.episode)
        self.tensorboard_logger.log_scalar('environment/changes_per_episode', env_stats['avg_changes_per_episode'], self.episode)
    
    def _print_progress(self):
        """Print training progress."""
        if len(self.episode_rewards) == 0:
            return
        
        avg_reward = np.mean(self.episode_rewards[-100:])
        avg_length = np.mean(self.episode_lengths[-100:])
        agent_stats = self.agent.get_stats()
        stage_info = self.curriculum.get_current_stage_info()
        
        elapsed_time = time.time() - self.training_start_time
        episodes_per_hour = self.episode / (elapsed_time / 3600)
        
        print(f"\nEpisode {self.episode}:")
        print(f"  Avg Reward (100): {avg_reward:.2f}")
        print(f"  Avg Length (100): {avg_length:.1f}")
        print(f"  Epsilon: {agent_stats['epsilon']:.3f}")
        print(f"  Buffer Size: {agent_stats['buffer_size']}")
        print(f"  Current Stage: {stage_info['name']} (Factor: {stage_info['difficulty_factor']:.1f})")
        print(f"  Episodes/Hour: {episodes_per_hour:.1f}")
        print(f"  Total Steps: {self.total_steps}")
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.save_dir / filename
        
        # Save agent
        self.agent.save(str(checkpoint_path))
        
        # Save additional training state
        training_state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_avg_reward': self.best_avg_reward,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_scores': self.evaluation_scores,
            'curriculum_state': {
                'current_stage': self.curriculum.current_stage,
                'episodes_in_stage': self.curriculum.episodes_in_stage,
                'stage_history': self.curriculum.get_stage_history()
            },
            'adaptation_state': self.adaptation_detector.get_state(),
            'config': self.config.to_dict()
        }
        
        torch.save(training_state, str(checkpoint_path.with_suffix('.training_state')))
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _finalize_training(self):
        """Finalize training and save results."""
        self.logger.info("Finalizing training...")
        
        # Save final model
        self._save_checkpoint('final_model.pth')
        
        # Export training data
        training_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_scores': self.evaluation_scores,
            'adaptation_events': self.agent.adaptation_events,
            'environmental_changes': self.agent.environmental_changes,
            'curriculum_history': self.curriculum.get_stage_history(),
            'difficulty_statistics': self.env.get_difficulty_statistics(),
            'agent_statistics': self.agent.get_stats()
        }
        
        # Save as numpy arrays for analysis
        np.save(self.save_dir / 'training_data.npy', training_data)
        
        # Generate final analysis
        self.performance_analyzer.analyze_training(
            training_data, 
            save_path=str(self.save_dir / 'analysis')
        )
        
        # Close loggers
        self.tensorboard_logger.close()
        
        self.logger.info(f"Training completed! Results saved to: {self.save_dir}")
        
        # Print final statistics
        total_time = time.time() - self.training_start_time
        print(f"\nTraining Summary:")
        print(f"  Total Episodes: {self.episode}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Training Time: {total_time / 3600:.1f} hours")
        print(f"  Best Avg Reward: {self.best_avg_reward:.2f}")
        print(f"  Final Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"  Adaptations: {len(self.agent.adaptation_events)}")
        print(f"  Env Changes: {len(self.agent.environmental_changes)}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Adaptive DQN Agent')
    parser.add_argument('--config', type=str, default='experiments/adaptive_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default='adaptive_breakout',
                       help='Name of the experiment')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = AdaptiveTrainer(
        config=config,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        # Implementation for resuming would go here
        print(f"Resuming from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
