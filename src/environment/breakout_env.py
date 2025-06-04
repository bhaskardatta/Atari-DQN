"""
Modified Atari Breakout environment with dynamic difficulty.
"""

import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from collections import deque

from .difficulty import DynamicDifficulty

# Try to register ALE environments on import
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass  # ALE not available, but may work with pre-installed envs


class BreakoutWrapper(gym.Wrapper):
    """
    Base wrapper for Breakout environment with preprocessing.
    Handles frame stacking, resizing, and grayscale conversion.
    """
    
    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_no_ops: int = 30,
        terminal_on_life_loss: bool = True
    ):
        """
        Initialize Breakout wrapper.
        
        Args:
            env: Base Gymnasium environment
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to skip between actions
            max_no_ops: Maximum number of no-op actions at start
            terminal_on_life_loss: Whether to terminate on life loss
        """
        super().__init__(env)
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_no_ops = max_no_ops
        self.terminal_on_life_loss = terminal_on_life_loss
        
        # Frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Observation space: stacked grayscale frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8
        )
        
        # Game state tracking
        self.lives = 0
        self.game_over = False
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        
        # Perform random number of no-op actions
        no_ops = np.random.randint(0, self.max_no_ops + 1)
        for _ in range(no_ops):
            obs, _, terminated, truncated, info = self.env.step(0)  # No-op action
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                break
        
        # Initialize frame stack
        processed_frame = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        # Initialize game state
        self.lives = info.get('lives', 0)
        self.game_over = False
        
        return self._get_observation(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action with frame skipping."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Execute action for frame_skip frames
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Process frame and update stack
        processed_frame = self._preprocess_frame(obs)
        self.frames.append(processed_frame)
        
        # Check for life loss
        if self.terminal_on_life_loss:
            current_lives = info.get('lives', 0)
            if current_lives < self.lives and current_lives > 0:
                # Life lost but game continues
                terminated = True
            self.lives = current_lives
        
        return self._get_observation(), total_reward, terminated, truncated, info
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame: convert to grayscale and resize.
        
        Args:
            frame: Raw frame from environment
        
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from stacked frames."""
        return np.array(self.frames, dtype=np.uint8)


class DynamicBreakoutEnv(BreakoutWrapper):
    """
    Breakout environment with dynamic difficulty modifications.
    
    Implements:
    - Variable paddle speed
    - Dynamic ball speed changes
    - Brick regeneration
    - Paddle size modifications
    """
    
    def __init__(
        self,
        env_name: str = "BreakoutNoFrameskip-v4",
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_no_ops: int = 30,
        terminal_on_life_loss: bool = True,
        difficulty_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize dynamic Breakout environment.
        
        Args:
            env_name: Name of base Atari environment
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to skip
            max_no_ops: Maximum no-op actions at start
            terminal_on_life_loss: Whether to terminate on life loss
            difficulty_config: Configuration for dynamic difficulty
            render_mode: Rendering mode for environment
        """
        # Create base environment
        base_env = gym.make(env_name, render_mode=render_mode)
        super().__init__(base_env, frame_stack, frame_skip, max_no_ops, terminal_on_life_loss)
        
        # Initialize dynamic difficulty system
        if difficulty_config is None:
            difficulty_config = {}
        
        # Map config parameters to DynamicDifficulty constructor parameters
        dynamic_difficulty_params = {}
        
        # Map parameter names from config to DynamicDifficulty constructor
        param_mapping = {
            'paddle_speed_range': 'paddle_speed_range',
            'ball_speed_range': 'ball_speed_multiplier_range',
            'paddle_size_range': 'paddle_size_range',
            'change_frequency': 'paddle_size_change_interval'
        }
        
        for config_key, constructor_key in param_mapping.items():
            if config_key in difficulty_config:
                value = difficulty_config[config_key]
                # Convert lists to tuples if needed
                if isinstance(value, list) and len(value) == 2:
                    value = tuple(value)
                dynamic_difficulty_params[constructor_key] = value
        
        # Set brick regeneration probability based on config
        if difficulty_config.get('brick_regeneration', False):
            dynamic_difficulty_params['brick_regen_prob'] = 0.001
        else:
            dynamic_difficulty_params['brick_regen_prob'] = 0.0
        
        # Set default values for unspecified parameters
        if 'ball_speed_change_prob' not in dynamic_difficulty_params:
            dynamic_difficulty_params['ball_speed_change_prob'] = 0.01
        
        self.difficulty = DynamicDifficulty(**dynamic_difficulty_params)
        
        # Game state tracking for difficulty
        self.previous_brick_count = 0
        self.destroyed_bricks = []
        self.episode_changes = []
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and difficulty system."""
        # Reset difficulty for new episode
        self.difficulty.reset_episode()
        
        # Reset game state tracking
        self.previous_brick_count = 0
        self.destroyed_bricks = []
        self.episode_changes = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Reset base environment
        obs, info = super().reset(**kwargs)
        
        # Add difficulty info
        difficulty_state = self.difficulty.update_frame()
        info.update({
            'difficulty': difficulty_state,
            'difficulty_changes': [],
            'brick_count': self._estimate_brick_count(obs)
        })
        
        self.previous_brick_count = info['brick_count']
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action with dynamic difficulty modifications."""
        # Execute base step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track episode statistics
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Detect destroyed bricks
        current_brick_count = self._estimate_brick_count(obs)
        if current_brick_count < self.previous_brick_count:
            bricks_destroyed = self.previous_brick_count - current_brick_count
            # Estimate destroyed brick positions (simplified)
            self.destroyed_bricks.extend([f"brick_{i}" for i in range(bricks_destroyed)])
        
        # Update difficulty system
        difficulty_state = self.difficulty.update_frame(self.destroyed_bricks)
        
        # Apply difficulty modifications (simulated)
        modified_reward = self._apply_difficulty_reward_modification(reward, difficulty_state)
        
        # Track changes made this frame
        frame_changes = difficulty_state.get('recent_changes', {})
        if frame_changes:
            self.episode_changes.append({
                'frame': self.difficulty.frame_count,
                'changes': frame_changes.copy()
            })
        
        # Update info with difficulty information
        info.update({
            'difficulty': difficulty_state,
            'difficulty_changes': frame_changes,
            'brick_count': current_brick_count,
            'original_reward': reward,
            'modified_reward': modified_reward,
            'episode_changes': len(self.episode_changes)
        })
        
        # Update tracking variables
        self.previous_brick_count = current_brick_count
        self.destroyed_bricks = []  # Reset for next frame
        
        # Record episode statistics if episode ended
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Add episode summary to info
            info.update({
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'episode_difficulty_changes': len(self.episode_changes),
                'difficulty_statistics': self.difficulty.get_statistics()
            })
        
        return obs, modified_reward, terminated, truncated, info
    
    def _estimate_brick_count(self, observation: np.ndarray) -> int:
        """
        Estimate number of bricks remaining from observation.
        
        This is a simplified implementation that counts colored pixels
        in the brick region of the screen.
        
        Args:
            observation: Current observation
        
        Returns:
            Estimated number of bricks
        """
        # Use the most recent frame from the stack
        frame = observation[-1] if len(observation.shape) == 3 else observation
        
        # Define brick region (approximate)
        brick_region = frame[15:35, :]  # Upper portion where bricks are located
        
        # Count non-zero pixels (simplified brick detection)
        brick_pixels = np.sum(brick_region > 50)  # Threshold for colored pixels
        
        # Estimate brick count (very rough approximation)
        estimated_bricks = brick_pixels // 50  # Assume ~50 pixels per brick
        
        return max(0, estimated_bricks)
    
    def _apply_difficulty_reward_modification(
        self, 
        reward: float, 
        difficulty_state: Dict[str, Any]
    ) -> float:
        """
        Apply reward modifications based on current difficulty.
        
        This can be used to provide additional learning signals
        or to compensate for difficulty changes.
        
        Args:
            reward: Original reward
            difficulty_state: Current difficulty state
        
        Returns:
            Modified reward
        """
        modified_reward = reward
        
        # Example modifications:
        
        # Bonus reward for adapting to paddle size changes
        if 'paddle_size' in difficulty_state.get('recent_changes', {}):
            size_change = difficulty_state['recent_changes']['paddle_size']
            if size_change < 0.9:  # Smaller paddle
                modified_reward += 0.1  # Small bonus for harder condition
        
        # Penalty for ball speed increases (to encourage adaptation)
        if 'ball_speed' in difficulty_state.get('recent_changes', {}):
            speed_change = difficulty_state['recent_changes']['ball_speed']
            if speed_change > 1.2:  # Faster ball
                modified_reward *= 0.95  # Slight penalty
        
        # Bonus for brick regeneration (more challenging)
        if 'brick_regen' in difficulty_state.get('recent_changes', {}):
            regen_count = len(difficulty_state['recent_changes']['brick_regen'])
            if regen_count > 0:
                modified_reward += 0.05 * regen_count
        
        return modified_reward
    
    def set_difficulty_factor(self, factor: float):
        """Set overall difficulty factor."""
        self.difficulty.set_difficulty_factor(factor)
    
    def get_difficulty_statistics(self) -> Dict[str, Any]:
        """Get comprehensive difficulty statistics."""
        base_stats = self.difficulty.get_statistics()
        
        # Add environment-specific statistics
        env_stats = {
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'recent_performance': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
        
        base_stats.update(env_stats)
        return base_stats
    
    def get_change_history(self) -> list:
        """Get complete history of difficulty changes."""
        return self.difficulty.get_change_history()
    
    def export_episode_data(self) -> Dict[str, Any]:
        """Export episode data for analysis."""
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'difficulty_changes': self.difficulty.export_change_data(),
            'difficulty_statistics': self.get_difficulty_statistics()
        }


def make_dynamic_breakout(
    difficulty_factor: float = 1.0,
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
    **difficulty_kwargs
) -> DynamicBreakoutEnv:
    """
    Factory function to create DynamicBreakoutEnv with specified difficulty.
    
    Args:
        difficulty_factor: Overall difficulty scaling (0.0 to 1.0)
        frame_stack: Number of frames to stack
        render_mode: Rendering mode
        **difficulty_kwargs: Additional difficulty configuration
    
    Returns:
        Configured DynamicBreakoutEnv
    """
    difficulty_config = {
        'difficulty_factor': difficulty_factor,
        **difficulty_kwargs
    }
    
    return DynamicBreakoutEnv(
        frame_stack=frame_stack,
        render_mode=render_mode,
        difficulty_config=difficulty_config
    )
