"""
Performance analysis and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for the adaptive RL agent.
    Generates plots and statistics for training analysis.
    """
    
    def __init__(self, save_plots: bool = True, plot_format: str = 'png'):
        """
        Initialize performance analyzer.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_format: Format for saved plots ('png', 'pdf', 'svg')
        """
        self.save_plots = save_plots
        self.plot_format = plot_format
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
    
    def analyze_training(self, training_data: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive training analysis.
        
        Args:
            training_data: Dictionary containing training data
            save_path: Optional path to save analysis results
        
        Returns:
            Dictionary with analysis results
        """
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {}
        
        # Basic training metrics
        analysis_results['basic_metrics'] = self._analyze_basic_metrics(training_data)
        
        # Learning curves
        self._plot_learning_curves(training_data, save_path)
        
        # Adaptation analysis
        analysis_results['adaptation_analysis'] = self._analyze_adaptation(training_data)
        self._plot_adaptation_analysis(training_data, save_path)
        
        # Curriculum learning analysis
        analysis_results['curriculum_analysis'] = self._analyze_curriculum(training_data)
        self._plot_curriculum_analysis(training_data, save_path)
        
        # Environmental change analysis
        analysis_results['environment_analysis'] = self._analyze_environment_changes(training_data)
        self._plot_environment_analysis(training_data, save_path)
        
        # Performance degradation and recovery
        analysis_results['recovery_analysis'] = self._analyze_recovery_patterns(training_data)
        self._plot_recovery_analysis(training_data, save_path)
        
        # Save analysis summary
        if save_path:
            self._save_analysis_summary(analysis_results, save_path)
        
        return analysis_results
    
    def _analyze_basic_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze basic training metrics."""
        episode_rewards = np.array(data['episode_rewards'])
        episode_lengths = np.array(data['episode_lengths'])
        
        # Handle empty arrays
        if len(episode_rewards) == 0:
            return {
                'total_episodes': 0,
                'final_performance': {
                    'mean_reward': 0.0,
                    'std_reward': 0.0,
                    'max_reward': 0.0,
                    'min_reward': 0.0
                },
                'episode_statistics': {
                    'mean_length': 0.0,
                    'std_length': 0.0,
                    'max_length': 0.0,
                    'min_length': 0.0
                },
                'learning_stability': {
                    'reward_variance': 0.0,
                    'coefficient_of_variation': 0.0
                }
            }
        
        return {
            'total_episodes': len(episode_rewards),
            'final_performance': {
                'mean_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.std(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards)
            },
            'episode_statistics': {
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'max_length': np.max(episode_lengths),
                'min_length': np.min(episode_lengths)
            },
            'learning_stability': {
                'reward_variance': np.var(episode_rewards),
                'coefficient_of_variation': np.std(episode_rewards) / (np.mean(episode_rewards) + 1e-8)
            }
        }
    
    def _plot_learning_curves(self, data: Dict[str, Any], save_path: Optional[Path]):
        """Plot learning curves."""
        episode_rewards = np.array(data['episode_rewards'])
        evaluation_scores = data.get('evaluation_scores', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
        
        # Episode rewards with moving average
        ax1 = axes[0, 0]
        episodes = np.arange(len(episode_rewards))
        ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Moving average
        window_size = min(100, len(episode_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(episodes[window_size-1:], moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Evaluation scores
        ax2 = axes[0, 1]
        if evaluation_scores:
            eval_episodes = np.arange(0, len(episode_rewards), len(episode_rewards) // len(evaluation_scores))[:len(evaluation_scores)]
            ax2.plot(eval_episodes, evaluation_scores, 'o-', color='green', linewidth=2, markersize=6)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Evaluation Score')
            ax2.set_title('Evaluation Performance')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No evaluation data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Evaluation Performance (No Data)')
        
        # Reward distribution
        ax3 = axes[1, 0]
        ax3.hist(episode_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning progress (rolling statistics)
        ax4 = axes[1, 1]
        window_size = min(50, len(episode_rewards) // 20)
        if window_size > 1:
            rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()
            rolling_std = pd.Series(episode_rewards).rolling(window=window_size).std()
            
            ax4.plot(episodes, rolling_mean, label='Rolling Mean', color='blue', linewidth=2)
            ax4.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                           alpha=0.3, color='blue', label='±1 Std Dev')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.set_title(f'Learning Progress (Rolling Window: {window_size})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path and self.save_plots:
            plt.savefig(save_path / f'learning_curves.{self.plot_format}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_adaptation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze adaptation events and performance."""
        adaptation_events = data.get('adaptation_events', [])
        environmental_changes = data.get('environmental_changes', [])
        
        if not adaptation_events and not environmental_changes:
            return {'message': 'No adaptation events recorded'}
        
        return {
            'total_adaptation_events': len(adaptation_events),
            'total_environmental_changes': len(environmental_changes),
            'adaptation_success_rate': len(adaptation_events) / max(1, len(environmental_changes)),
            'adaptation_statistics': {
                'mean_adaptation_time': np.mean([event.get('recovery_time', 0) for event in adaptation_events]) if adaptation_events else 0,
                'adaptation_times': [event.get('recovery_time', 0) for event in adaptation_events]
            }
        }
    
    def _plot_adaptation_analysis(self, data: Dict[str, Any], save_path: Optional[Path]):
        """Plot adaptation analysis."""
        adaptation_events = data.get('adaptation_events', [])
        environmental_changes = data.get('environmental_changes', [])
        episode_rewards = np.array(data['episode_rewards'])
        
        if not adaptation_events and not environmental_changes:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Adaptation Analysis', fontsize=16, fontweight='bold')
        
        # Performance around environmental changes
        ax1 = axes[0, 0]
        episodes = np.arange(len(episode_rewards))
        ax1.plot(episodes, episode_rewards, alpha=0.5, color='blue', label='Episode Reward')
        
        # Mark environmental changes
        for change in environmental_changes:
            change_episode = change.get('step', 0) // 1000  # Approximate episode from step
            if change_episode < len(episode_rewards):
                ax1.axvline(change_episode, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Mark adaptation events
        for event in adaptation_events:
            event_episode = event.get('step', 0) // 1000  # Approximate episode from step
            if event_episode < len(episode_rewards):
                ax1.axvline(event_episode, color='green', linestyle='-', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Performance Around Environmental Changes')
        ax1.legend(['Episode Reward', 'Environmental Change', 'Adaptation Event'])
        ax1.grid(True, alpha=0.3)
        
        # Adaptation time distribution
        ax2 = axes[0, 1]
        if adaptation_events:
            adaptation_times = [event.get('recovery_time', 0) for event in adaptation_events if event.get('recovery_time', 0) > 0]
            if adaptation_times:
                ax2.hist(adaptation_times, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(np.mean(adaptation_times), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(adaptation_times):.1f}')
                ax2.set_xlabel('Adaptation Time (episodes)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Adaptation Time Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No adaptation time data', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No adaptation events', ha='center', va='center', transform=ax2.transAxes)
        
        # Performance degradation and recovery
        ax3 = axes[1, 0]
        if environmental_changes:
            performance_drops = [change.get('performance_drop', 0) for change in environmental_changes]
            ax3.bar(range(len(performance_drops)), performance_drops, alpha=0.7, color='orange')
            ax3.set_xlabel('Environmental Change Event')
            ax3.set_ylabel('Performance Drop')
            ax3.set_title('Performance Degradation per Change')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No environmental changes', ha='center', va='center', transform=ax3.transAxes)
        
        # Adaptation success over time
        ax4 = axes[1, 1]
        if adaptation_events:
            adaptation_episodes = [event.get('step', 0) // 1000 for event in adaptation_events]
            adaptation_success = np.ones(len(adaptation_episodes))  # All recorded events are successful
            
            ax4.scatter(adaptation_episodes, adaptation_success, color='green', s=50, alpha=0.7, label='Successful Adaptation')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Adaptation Success')
            ax4.set_title('Adaptation Success Over Time')
            ax4.set_ylim(-0.1, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No adaptation events', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path and self.save_plots:
            plt.savefig(save_path / f'adaptation_analysis.{self.plot_format}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_curriculum(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze curriculum learning progression."""
        curriculum_history = data.get('curriculum_history', [])
        
        if not curriculum_history:
            return {'message': 'No curriculum learning data available'}
        
        return {
            'total_stages': len(curriculum_history),
            'stage_statistics': curriculum_history,
            'curriculum_effectiveness': {
                'stages_completed': len(curriculum_history),
                'average_stage_duration': np.mean([stage['episodes'] for stage in curriculum_history]) if curriculum_history else 0,
                'performance_progression': [stage['avg_performance'] for stage in curriculum_history]
            }
        }
    
    def _plot_curriculum_analysis(self, data: Dict[str, Any], save_path: Optional[Path]):
        """Plot curriculum learning analysis."""
        curriculum_history = data.get('curriculum_history', [])
        episode_rewards = np.array(data['episode_rewards'])
        
        if not curriculum_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Curriculum Learning Analysis', fontsize=16, fontweight='bold')
        
        # Performance progression through stages
        ax1 = axes[0, 0]
        stage_names = [stage['stage_name'] for stage in curriculum_history]
        stage_performances = [stage['avg_performance'] for stage in curriculum_history]
        
        ax1.plot(range(len(stage_performances)), stage_performances, 'o-', linewidth=2, markersize=8, color='purple')
        ax1.set_xlabel('Curriculum Stage')
        ax1.set_ylabel('Average Performance')
        ax1.set_title('Performance Progression Through Curriculum')
        ax1.set_xticks(range(len(stage_names)))
        ax1.set_xticklabels(stage_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Stage duration analysis
        ax2 = axes[0, 1]
        stage_durations = [stage['episodes'] for stage in curriculum_history]
        ax2.bar(range(len(stage_durations)), stage_durations, alpha=0.7, color='teal')
        ax2.set_xlabel('Curriculum Stage')
        ax2.set_ylabel('Episodes in Stage')
        ax2.set_title('Time Spent in Each Stage')
        ax2.set_xticks(range(len(stage_names)))
        ax2.set_xticklabels(stage_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Cumulative performance over curriculum
        ax3 = axes[1, 0]
        episodes = np.arange(len(episode_rewards))
        ax3.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Mark stage transitions
        cumulative_episodes = 0
        for i, stage in enumerate(curriculum_history):
            cumulative_episodes += stage['episodes']
            if cumulative_episodes < len(episode_rewards):
                ax3.axvline(cumulative_episodes, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax3.text(cumulative_episodes, ax3.get_ylim()[1] * 0.9, stage['stage_name'], 
                        rotation=90, ha='right', va='top', fontsize=10)
        
        # Moving average
        window_size = min(50, len(episode_rewards) // 20)
        if window_size > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(episodes[window_size-1:], moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.set_title('Performance Throughout Curriculum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Stage efficiency (performance gain per episode)
        ax4 = axes[1, 1]
        if len(curriculum_history) > 1:
            stage_efficiency = []
            for i in range(1, len(curriculum_history)):
                current_perf = curriculum_history[i]['avg_performance']
                prev_perf = curriculum_history[i-1]['avg_performance']
                episodes_in_stage = curriculum_history[i]['episodes']
                efficiency = (current_perf - prev_perf) / episodes_in_stage if episodes_in_stage > 0 else 0
                stage_efficiency.append(efficiency)
            
            stage_labels = [stage['stage_name'] for stage in curriculum_history[1:]]
            ax4.bar(range(len(stage_efficiency)), stage_efficiency, alpha=0.7, color='orange')
            ax4.set_xlabel('Curriculum Stage')
            ax4.set_ylabel('Performance Gain per Episode')
            ax4.set_title('Learning Efficiency by Stage')
            ax4.set_xticks(range(len(stage_labels)))
            ax4.set_xticklabels(stage_labels, rotation=45)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient stages for efficiency analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path and self.save_plots:
            plt.savefig(save_path / f'curriculum_analysis.{self.plot_format}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_environment_changes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental changes and their impact."""
        difficulty_stats = data.get('difficulty_statistics', {})
        
        return {
            'total_changes': difficulty_stats.get('total_changes', 0),
            'changes_by_factor': difficulty_stats.get('changes_by_factor', {}),
            'average_changes_per_episode': difficulty_stats.get('avg_changes_per_episode', 0.0),
            'current_difficulty': difficulty_stats.get('current_difficulty_factor', 0.0)
        }
    
    def _plot_environment_analysis(self, data: Dict[str, Any], save_path: Optional[Path]):
        """Plot environmental change analysis."""
        difficulty_stats = data.get('difficulty_statistics', {})
        
        if not difficulty_stats or difficulty_stats.get('total_changes', 0) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Environmental Changes Analysis', fontsize=16, fontweight='bold')
        
        # Changes by factor type
        ax1 = axes[0, 0]
        changes_by_factor = difficulty_stats.get('changes_by_factor', {})
        if changes_by_factor:
            factors = list(changes_by_factor.keys())
            counts = list(changes_by_factor.values())
            
            ax1.pie(counts, labels=factors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribution of Environmental Changes by Factor')
        else:
            ax1.text(0.5, 0.5, 'No change data by factor', ha='center', va='center', transform=ax1.transAxes)
        
        # Change frequency over time (if available)
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, 'Change frequency over time\n(requires detailed change history)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Change Frequency Over Time')
        
        # Impact on performance (placeholder)
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, 'Performance impact analysis\n(requires detailed correlation data)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Impact of Changes on Performance')
        
        # Difficulty progression
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Difficulty progression\n(requires episode-level difficulty data)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Difficulty Progression Over Time')
        
        plt.tight_layout()
        
        if save_path and self.save_plots:
            plt.savefig(save_path / f'environment_analysis.{self.plot_format}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_recovery_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance recovery patterns after changes."""
        episode_rewards = np.array(data['episode_rewards'])
        environmental_changes = data.get('environmental_changes', [])
        
        if not environmental_changes:
            return {'message': 'No environmental changes to analyze recovery patterns'}
        
        recovery_patterns = []
        
        for change in environmental_changes:
            change_episode = change.get('step', 0) // 1000  # Approximate episode
            
            if change_episode < len(episode_rewards) - 50:  # Need at least 50 episodes after change
                pre_change_performance = np.mean(episode_rewards[max(0, change_episode-20):change_episode])
                post_change_performance = episode_rewards[change_episode:change_episode+50]
                
                recovery_pattern = {
                    'change_episode': change_episode,
                    'pre_change_performance': pre_change_performance,
                    'immediate_performance': np.mean(post_change_performance[:10]),
                    'recovery_performance': np.mean(post_change_performance[40:50]),
                    'recovery_time': self._estimate_recovery_time(post_change_performance, pre_change_performance)
                }
                recovery_patterns.append(recovery_pattern)
        
        return {
            'total_recoveries_analyzed': len(recovery_patterns),
            'average_recovery_time': np.mean([p['recovery_time'] for p in recovery_patterns]) if recovery_patterns else 0,
            'recovery_patterns': recovery_patterns
        }
    
    def _estimate_recovery_time(self, post_change_performance: np.ndarray, pre_change_baseline: float) -> int:
        """Estimate time to recover to baseline performance."""
        threshold = pre_change_baseline * 0.9  # 90% of baseline
        
        for i, performance in enumerate(post_change_performance):
            if performance >= threshold:
                return i
        
        return len(post_change_performance)  # Didn't recover within window
    
    def _plot_recovery_analysis(self, data: Dict[str, Any], save_path: Optional[Path]):
        """Plot recovery pattern analysis."""
        # This would contain detailed recovery analysis plots
        # For now, create a placeholder
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Recovery Patterns Analysis', fontsize=16, fontweight='bold')
        
        ax.text(0.5, 0.5, 'Recovery patterns analysis\n(detailed implementation needed)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Performance Recovery After Environmental Changes')
        
        if save_path and self.save_plots:
            plt.savefig(save_path / f'recovery_analysis.{self.plot_format}', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _save_analysis_summary(self, analysis_results: Dict[str, Any], save_path: Path):
        """Save analysis summary to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(analysis_results)
        
        with open(save_path / 'analysis_summary.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis summary saved to: {save_path / 'analysis_summary.json'}")
