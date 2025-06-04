"""
Dynamic difficulty system for Atari Breakout.
"""

import numpy as np
import random
from typing import Dict, Any, Optional
from enum import Enum


class DifficultyFactor(Enum):
    """Types of difficulty modifications."""
    PADDLE_SPEED = "paddle_speed"
    BALL_SPEED = "ball_speed"
    BRICK_REGENERATION = "brick_regen"
    PADDLE_SIZE = "paddle_size"


class DynamicDifficulty:
    """
    Dynamic difficulty system that modifies game parameters during gameplay.
    
    Modifications include:
    - Paddle speed variation (50% to 150% of normal)
    - Ball speed increases during episode
    - Brick regeneration
    - Paddle size changes every 500 frames
    """
    
    def __init__(
        self,
        paddle_speed_range: tuple = (0.5, 1.5),
        ball_speed_multiplier_range: tuple = (1.0, 2.0),
        ball_speed_change_prob: float = 0.01,
        brick_regen_prob: float = 0.001,
        paddle_size_change_interval: int = 500,
        paddle_size_range: tuple = (0.7, 1.3),
        difficulty_factor: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize dynamic difficulty system.
        
        Args:
            paddle_speed_range: Min and max paddle speed multipliers
            ball_speed_multiplier_range: Min and max ball speed multipliers
            ball_speed_change_prob: Probability of ball speed change per frame
            brick_regen_prob: Probability of brick regeneration per frame
            paddle_size_change_interval: Frames between paddle size changes
            paddle_size_range: Min and max paddle size multipliers
            difficulty_factor: Overall difficulty scaling (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.paddle_speed_range = paddle_speed_range
        self.ball_speed_multiplier_range = ball_speed_multiplier_range
        self.ball_speed_change_prob = ball_speed_change_prob
        self.brick_regen_prob = brick_regen_prob
        self.paddle_size_change_interval = paddle_size_change_interval
        self.paddle_size_range = paddle_size_range
        self.difficulty_factor = difficulty_factor
        
        # Current state
        self.frame_count = 0
        self.episode_count = 0
        self.current_paddle_speed = 1.0
        self.current_ball_speed_multiplier = 1.0
        self.current_paddle_size = 1.0
        
        # Change history
        self.change_history = []
        self.active_changes = {}
        
        # Random state
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.reset_episode()
    
    def reset_episode(self):
        """Reset difficulty state for new episode."""
        self.frame_count = 0
        self.episode_count += 1
        
        # Reset to base values
        self.current_paddle_speed = 1.0
        self.current_ball_speed_multiplier = 1.0
        self.current_paddle_size = 1.0
        
        # Clear active changes
        self.active_changes = {}
        
        # Apply initial difficulty based on difficulty factor
        if self.difficulty_factor > 0:
            self._apply_initial_difficulty()
    
    def _apply_initial_difficulty(self):
        """Apply initial difficulty modifications based on difficulty factor."""
        # Scale probabilities and ranges by difficulty factor
        scaled_factor = self.difficulty_factor
        
        # Initial paddle speed variation
        if random.random() < 0.3 * scaled_factor:
            speed_range = (
                1.0 - (1.0 - self.paddle_speed_range[0]) * scaled_factor,
                1.0 + (self.paddle_speed_range[1] - 1.0) * scaled_factor
            )
            self.current_paddle_speed = random.uniform(*speed_range)
            self._log_change(DifficultyFactor.PADDLE_SPEED, self.current_paddle_speed)
        
        # Initial ball speed
        if random.random() < 0.2 * scaled_factor:
            speed_range = (
                1.0,
                1.0 + (self.ball_speed_multiplier_range[1] - 1.0) * scaled_factor
            )
            self.current_ball_speed_multiplier = random.uniform(*speed_range)
            self._log_change(DifficultyFactor.BALL_SPEED, self.current_ball_speed_multiplier)
    
    def update_frame(self, destroyed_bricks: Optional[list] = None) -> Dict[str, Any]:
        """
        Update difficulty for current frame.
        
        Args:
            destroyed_bricks: List of recently destroyed brick positions
        
        Returns:
            Dictionary of current difficulty modifiers
        """
        self.frame_count += 1
        changes_made = {}
        
        if self.difficulty_factor == 0:
            # No difficulty modifications
            return self._get_current_state()
        
        # Paddle speed changes (random variation)
        if random.random() < 0.001 * self.difficulty_factor:
            speed_range = (
                1.0 - (1.0 - self.paddle_speed_range[0]) * self.difficulty_factor,
                1.0 + (self.paddle_speed_range[1] - 1.0) * self.difficulty_factor
            )
            new_speed = random.uniform(*speed_range)
            if abs(new_speed - self.current_paddle_speed) > 0.1:
                self.current_paddle_speed = new_speed
                changes_made[DifficultyFactor.PADDLE_SPEED.value] = new_speed
                self._log_change(DifficultyFactor.PADDLE_SPEED, new_speed)
        
        # Ball speed increases (occasional mid-episode)
        scaled_prob = self.ball_speed_change_prob * self.difficulty_factor
        if random.random() < scaled_prob:
            max_multiplier = 1.0 + (self.ball_speed_multiplier_range[1] - 1.0) * self.difficulty_factor
            new_multiplier = min(max_multiplier, self.current_ball_speed_multiplier * 1.1)
            if new_multiplier != self.current_ball_speed_multiplier:
                self.current_ball_speed_multiplier = new_multiplier
                changes_made[DifficultyFactor.BALL_SPEED.value] = new_multiplier
                self._log_change(DifficultyFactor.BALL_SPEED, new_multiplier)
        
        # Brick regeneration
        regenerated_bricks = []
        if destroyed_bricks and self.difficulty_factor > 0:
            scaled_regen_prob = self.brick_regen_prob * self.difficulty_factor
            for brick_pos in destroyed_bricks:
                if random.random() < scaled_regen_prob:
                    regenerated_bricks.append(brick_pos)
            
            if regenerated_bricks:
                changes_made[DifficultyFactor.BRICK_REGENERATION.value] = regenerated_bricks
                self._log_change(DifficultyFactor.BRICK_REGENERATION, len(regenerated_bricks))
        
        # Paddle size changes (every N frames)
        if self.frame_count % self.paddle_size_change_interval == 0 and self.difficulty_factor > 0:
            size_range = (
                1.0 - (1.0 - self.paddle_size_range[0]) * self.difficulty_factor,
                1.0 + (self.paddle_size_range[1] - 1.0) * self.difficulty_factor
            )
            new_size = random.uniform(*size_range)
            if abs(new_size - self.current_paddle_size) > 0.05:
                self.current_paddle_size = new_size
                changes_made[DifficultyFactor.PADDLE_SIZE.value] = new_size
                self._log_change(DifficultyFactor.PADDLE_SIZE, new_size)
        
        # Update active changes
        self.active_changes.update(changes_made)
        
        return self._get_current_state()
    
    def _log_change(self, factor: DifficultyFactor, value: float):
        """Log a difficulty change."""
        self.change_history.append({
            'episode': self.episode_count,
            'frame': self.frame_count,
            'factor': factor.value,
            'value': value,
            'difficulty_level': self.difficulty_factor
        })
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current difficulty state."""
        return {
            'paddle_speed_multiplier': self.current_paddle_speed,
            'ball_speed_multiplier': self.current_ball_speed_multiplier,
            'paddle_size_multiplier': self.current_paddle_size,
            'difficulty_factor': self.difficulty_factor,
            'frame_count': self.frame_count,
            'episode_count': self.episode_count,
            'recent_changes': dict(self.active_changes)
        }
    
    def set_difficulty_factor(self, factor: float):
        """Set overall difficulty factor (0.0 to 1.0)."""
        self.difficulty_factor = max(0.0, min(1.0, factor))
        self._log_change(DifficultyFactor.PADDLE_SPEED, self.difficulty_factor)  # Use as general marker
    
    def get_change_history(self) -> list:
        """Get history of all difficulty changes."""
        return self.change_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get difficulty statistics."""
        if not self.change_history:
            return {
                'total_changes': 0,
                'changes_by_factor': {},
                'avg_changes_per_episode': 0.0
            }
        
        # Count changes by factor
        changes_by_factor = {}
        for change in self.change_history:
            factor = change['factor']
            changes_by_factor[factor] = changes_by_factor.get(factor, 0) + 1
        
        return {
            'total_changes': len(self.change_history),
            'changes_by_factor': changes_by_factor,
            'avg_changes_per_episode': len(self.change_history) / max(1, self.episode_count),
            'current_difficulty_factor': self.difficulty_factor,
            'current_state': self._get_current_state()
        }
    
    def export_change_data(self) -> list:
        """Export change history for analysis."""
        return [
            {
                'episode': change['episode'],
                'frame': change['frame'],
                'factor': change['factor'],
                'value': change['value'],
                'difficulty_level': change['difficulty_level']
            }
            for change in self.change_history
        ]


class CurriculumDifficulty:
    """
    Curriculum learning difficulty scheduler.
    Gradually introduces environmental modifications.
    """
    
    def __init__(
        self,
        stages: list = None,
        stage_duration: int = 2000,
        performance_threshold: float = 0.7,
        patience: int = 5
    ):
        """
        Initialize curriculum difficulty scheduler.
        
        Args:
            stages: List of difficulty stages
            stage_duration: Episodes per stage
            performance_threshold: Performance threshold to advance
            patience: Evaluations before advancing regardless
        """
        if stages is None:
            stages = [
                {"name": "normal", "difficulty_factor": 0.0},
                {"name": "easy_dynamic", "difficulty_factor": 0.3},
                {"name": "medium_dynamic", "difficulty_factor": 0.6},
                {"name": "full_dynamic", "difficulty_factor": 1.0}
            ]
        
        self.stages = stages
        self.stage_duration = stage_duration
        self.performance_threshold = performance_threshold
        self.patience = patience
        
        # Current state
        self.current_stage = 0
        self.episodes_in_stage = 0
        self.evaluations_without_progress = 0
        self.stage_history = []
        
        # Performance tracking
        self.recent_performances = []
        self.performance_window = 100
    
    def update_episode(self, episode_reward: float) -> float:
        """
        Update curriculum based on episode performance.
        
        Args:
            episode_reward: Reward from completed episode
        
        Returns:
            Current difficulty factor
        """
        self.episodes_in_stage += 1
        self.recent_performances.append(episode_reward)
        
        # Keep only recent performances
        if len(self.recent_performances) > self.performance_window:
            self.recent_performances.pop(0)
        
        # Check if ready to advance stage
        if self._should_advance_stage():
            self._advance_stage()
        
        return self.get_current_difficulty_factor()
    
    def evaluate_performance(self, avg_performance: float) -> bool:
        """
        Evaluate current performance and decide whether to advance.
        
        Args:
            avg_performance: Average performance over evaluation period
        
        Returns:
            True if stage advanced
        """
        if avg_performance >= self.performance_threshold:
            self._advance_stage()
            return True
        else:
            self.evaluations_without_progress += 1
            
            # Advance anyway if stuck too long
            if self.evaluations_without_progress >= self.patience:
                self._advance_stage()
                return True
        
        return False
    
    def _should_advance_stage(self) -> bool:
        """Check if should advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        # Advance if duration reached
        if self.episodes_in_stage >= self.stage_duration:
            return True
        
        # Advance if performance threshold reached
        if len(self.recent_performances) >= self.performance_window:
            avg_performance = np.mean(self.recent_performances)
            normalized_performance = (avg_performance + 21) / 21  # Normalize Breakout scores
            return normalized_performance >= self.performance_threshold
        
        return False
    
    def _advance_stage(self):
        """Advance to next difficulty stage."""
        if self.current_stage < len(self.stages) - 1:
            # Log stage completion
            self.stage_history.append({
                'stage': self.current_stage,
                'stage_name': self.stages[self.current_stage]['name'],
                'episodes': self.episodes_in_stage,
                'avg_performance': np.mean(self.recent_performances) if self.recent_performances else 0.0
            })
            
            # Advance stage
            self.current_stage += 1
            self.episodes_in_stage = 0
            self.evaluations_without_progress = 0
            
            print(f"Advanced to stage {self.current_stage}: {self.stages[self.current_stage]['name']}")
    
    def get_current_difficulty_factor(self) -> float:
        """Get current difficulty factor."""
        return self.stages[self.current_stage]['difficulty_factor']
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        current_stage_info = self.stages[self.current_stage].copy()
        current_stage_info.update({
            'stage_index': self.current_stage,
            'episodes_in_stage': self.episodes_in_stage,
            'progress': self.episodes_in_stage / self.stage_duration,
            'recent_avg_performance': np.mean(self.recent_performances) if self.recent_performances else 0.0
        })
        return current_stage_info
    
    def get_stage_history(self) -> list:
        """Get history of completed stages."""
        return self.stage_history.copy()
    
    def force_stage(self, stage_index: int):
        """Force advancement to specific stage."""
        if 0 <= stage_index < len(self.stages):
            self.current_stage = stage_index
            self.episodes_in_stage = 0
            self.evaluations_without_progress = 0
