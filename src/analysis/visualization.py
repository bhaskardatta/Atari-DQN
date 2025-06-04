"""
Video recording and advanced visualization utilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import gymnasium as gym
from collections import deque
import logging


class VideoRecorder:
    """
    Records gameplay videos for analysis and demonstration.
    """
    
    def __init__(self, fps: int = 30, codec: str = 'mp4v'):
        """
        Initialize video recorder.
        
        Args:
            fps: Frames per second for recording
            codec: Video codec to use
        """
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frames = []
        self.recording = False
        
    def start_recording(self, filename: str, frame_size: Tuple[int, int]):
        """
        Start recording video.
        
        Args:
            filename: Output filename
            frame_size: Frame dimensions (width, height)
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, frame_size)
        self.frames = []
        self.recording = True
        logging.info(f"Started recording video: {filename}")
    
    def add_frame(self, frame: np.ndarray):
        """
        Add frame to recording.
        
        Args:
            frame: Frame to add (RGB format)
        """
        if self.recording and self.writer is not None:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr_frame)
            self.frames.append(frame.copy())
    
    def stop_recording(self):
        """Stop recording and save video."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.recording = False
        logging.info(f"Stopped recording. Saved {len(self.frames)} frames")
    
    def get_frames(self) -> List[np.ndarray]:
        """Get recorded frames."""
        return self.frames.copy()


class GameplayVisualizer:
    """
    Creates visualizations of gameplay and agent behavior.
    """
    
    def __init__(self):
        """Initialize gameplay visualizer."""
        self.action_names = {
            0: 'NOOP',
            1: 'FIRE',
            2: 'RIGHT',
            3: 'LEFT'
        }
    
    def create_action_heatmap(self, actions: List[int], episode_length: int) -> plt.Figure:
        """
        Create heatmap of action distribution over time.
        
        Args:
            actions: List of actions taken
            episode_length: Length of episode
            
        Returns:
            matplotlib Figure
        """
        # Bin actions by time segments
        num_bins = min(50, len(actions) // 10)
        if num_bins == 0:
            num_bins = 1
            
        bin_size = len(actions) // num_bins
        action_bins = np.zeros((num_bins, 4))  # 4 actions
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(actions))
            bin_actions = actions[start_idx:end_idx]
            
            for action in bin_actions:
                if 0 <= action < 4:
                    action_bins[i, action] += 1
        
        # Normalize
        action_bins = action_bins / (action_bins.sum(axis=1, keepdims=True) + 1e-8)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(action_bins.T, aspect='auto', cmap='viridis')
        
        ax.set_xlabel('Time Segment')
        ax.set_ylabel('Action')
        ax.set_title('Action Distribution Heatmap')
        ax.set_yticks(range(4))
        ax.set_yticklabels([self.action_names[i] for i in range(4)])
        
        plt.colorbar(im, ax=ax, label='Action Frequency')
        plt.tight_layout()
        
        return fig
    
    def create_q_value_visualization(self, q_values: List[np.ndarray], 
                                   actions: List[int]) -> plt.Figure:
        """
        Visualize Q-values over time.
        
        Args:
            q_values: List of Q-value arrays
            actions: List of actions taken
            
        Returns:
            matplotlib Figure
        """
        if not q_values:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Q-values available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        q_array = np.array(q_values)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot Q-values for each action
        for action in range(q_array.shape[1]):
            ax1.plot(q_array[:, action], label=self.action_names.get(action, f'Action {action}'))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Q-Value')
        ax1.set_title('Q-Values Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot max Q-value and selected actions
        max_q = np.max(q_array, axis=1)
        ax2.plot(max_q, label='Max Q-Value', color='red', linewidth=2)
        
        # Add action markers
        action_colors = ['blue', 'green', 'orange', 'purple']
        for i, action in enumerate(actions[:len(q_values)]):
            if 0 <= action < len(action_colors):
                ax2.scatter(i, max_q[i], color=action_colors[action], 
                           alpha=0.6, s=20)
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Max Q-Value')
        ax2.set_title('Max Q-Value and Selected Actions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_exploration_visualization(self, exploration_rates: List[float]) -> plt.Figure:
        """
        Visualize exploration rate over time.
        
        Args:
            exploration_rates: List of exploration rates (epsilon values)
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(exploration_rates, linewidth=2, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Rate (ε)')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key milestones
        if len(exploration_rates) > 0:
            max_eps = max(exploration_rates)
            min_eps = min(exploration_rates)
            
            ax.axhline(y=max_eps, color='red', linestyle='--', alpha=0.7, 
                      label=f'Max ε: {max_eps:.3f}')
            ax.axhline(y=min_eps, color='green', linestyle='--', alpha=0.7,
                      label=f'Min ε: {min_eps:.3f}')
            
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_adaptation_timeline(self, adaptation_events: List[Dict[str, Any]]) -> plt.Figure:
        """
        Create timeline visualization of adaptation events.
        
        Args:
            adaptation_events: List of adaptation event dictionaries
            
        Returns:
            matplotlib Figure
        """
        if not adaptation_events:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No adaptation events recorded', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract data
        episodes = [event.get('episode', 0) for event in adaptation_events]
        event_types = [event.get('type', 'unknown') for event in adaptation_events]
        confidences = [event.get('confidence', 0.5) for event in adaptation_events]
        
        # Create scatter plot with different colors for event types
        unique_types = list(set(event_types))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        
        for i, event_type in enumerate(unique_types):
            mask = np.array(event_types) == event_type
            episodes_filtered = np.array(episodes)[mask]
            confidences_filtered = np.array(confidences)[mask]
            
            ax.scatter(episodes_filtered, [i] * len(episodes_filtered),
                      s=confidences_filtered * 200 + 50,
                      c=[colors[i]], label=event_type, alpha=0.7)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Event Type')
        ax.set_title('Adaptation Events Timeline')
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_comparison(self, 
                                   performance_data: Dict[str, List[float]]) -> plt.Figure:
        """
        Create comparison plot of performance across different conditions.
        
        Args:
            performance_data: Dictionary with condition names as keys and performance lists as values
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        data_for_box = []
        labels = []
        for condition, values in performance_data.items():
            data_for_box.append(values)
            labels.append(condition)
        
        if data_for_box:
            ax1.boxplot(data_for_box, labels=labels)
            ax1.set_title('Performance Distribution by Condition')
            ax1.set_ylabel('Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # Line plot
        for condition, values in performance_data.items():
            ax2.plot(values, label=condition, alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class EnvironmentVisualizer:
    """
    Visualizes environment state and dynamics.
    """
    
    def __init__(self):
        """Initialize environment visualizer."""
        pass
    
    def create_difficulty_progression(self, difficulty_history: List[Dict[str, Any]]) -> plt.Figure:
        """
        Visualize difficulty progression over time.
        
        Args:
            difficulty_history: List of difficulty state dictionaries
            
        Returns:
            matplotlib Figure
        """
        if not difficulty_history:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No difficulty data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Extract parameters
        episodes = list(range(len(difficulty_history)))
        paddle_speeds = [d.get('paddle_speed_multiplier', 1.0) for d in difficulty_history]
        ball_speeds = [d.get('ball_speed_multiplier', 1.0) for d in difficulty_history]
        paddle_sizes = [d.get('paddle_size_multiplier', 1.0) for d in difficulty_history]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Paddle speed
        axes[0].plot(episodes, paddle_speeds, linewidth=2, color='blue')
        axes[0].set_ylabel('Paddle Speed Multiplier')
        axes[0].set_title('Difficulty Parameters Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Ball speed
        axes[1].plot(episodes, ball_speeds, linewidth=2, color='red')
        axes[1].set_ylabel('Ball Speed Multiplier')
        axes[1].grid(True, alpha=0.3)
        
        # Paddle size
        axes[2].plot(episodes, paddle_sizes, linewidth=2, color='green')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Paddle Size Multiplier')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_state_distribution(self, states: List[np.ndarray]) -> plt.Figure:
        """
        Visualize distribution of environment states.
        
        Args:
            states: List of environment states
            
        Returns:
            matplotlib Figure
        """
        if not states:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No state data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        # Sample states for visualization
        sample_size = min(100, len(states))
        sampled_states = states[::len(states)//sample_size]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Show sample states
        for i, ax in enumerate(axes.flat):
            if i < len(sampled_states):
                state = sampled_states[i]
                if len(state.shape) == 3 and state.shape[2] == 3:
                    ax.imshow(state)
                elif len(state.shape) == 2:
                    ax.imshow(state, cmap='gray')
                else:
                    # Flatten and show as 1D plot
                    ax.plot(state.flatten())
                ax.set_title(f'State {i+1}')
                ax.axis('off')
        
        plt.suptitle('Sample Environment States')
        plt.tight_layout()
        return fig


def save_visualization(fig: plt.Figure, filepath: str, format: str = 'png', 
                      dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save matplotlib figure with proper formatting.
    
    Args:
        fig: Figure to save
        filepath: Output file path
        format: File format
        dpi: Resolution for raster formats
        bbox_inches: Bounding box setting
    """
    fig.savefig(filepath, format=format, dpi=dpi, bbox_inches=bbox_inches)
    logging.info(f"Saved visualization: {filepath}")
