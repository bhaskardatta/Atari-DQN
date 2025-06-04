"""
Adaptation detection system for environmental changes.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from scipy import stats


class AdaptationDetector:
    """
    Detects environmental changes and tracks agent adaptation.
    
    Uses statistical methods to identify performance drops and recovery patterns
    that indicate environmental shifts and successful adaptation.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 0.2,
        min_episodes: int = 50,
        adaptation_threshold: float = 0.8
    ):
        """
        Initialize adaptation detector.
        
        Args:
            window_size: Size of rolling window for performance tracking
            sensitivity: Sensitivity to performance changes (0.0 to 1.0)
            min_episodes: Minimum episodes before detection starts
            adaptation_threshold: Threshold for considering adaptation successful
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_episodes = min_episodes
        self.adaptation_threshold = adaptation_threshold
        
        # Performance tracking
        self.performance_history = deque(maxlen=window_size * 3)
        self.difficulty_history = deque(maxlen=window_size * 3)
        
        # Change detection state
        self.baseline_performance = None
        self.recent_changes = []
        self.adaptation_events = []
        
        # Statistics
        self.total_changes_detected = 0
        self.successful_adaptations = 0
        self.adaptation_times = []
    
    def update(self, episode_reward: float, difficulty_level: float) -> Dict[str, Any]:
        """
        Update detector with new episode data.
        
        Args:
            episode_reward: Reward from completed episode
            difficulty_level: Current difficulty level
        
        Returns:
            Dictionary with detection results and adaptation info
        """
        # Add to history
        self.performance_history.append(episode_reward)
        self.difficulty_history.append(difficulty_level)
        
        # Initialize baseline if enough data
        if self.baseline_performance is None and len(self.performance_history) >= self.min_episodes:
            self.baseline_performance = np.mean(list(self.performance_history)[:self.min_episodes])
        
        # Detect changes if we have enough data
        change_detected = False
        change_type = None
        adaptation_score = 0.0
        recovery_time = None
        
        if len(self.performance_history) >= self.window_size:
            change_info = self._detect_environmental_change()
            if change_info['detected']:
                change_detected = True
                change_type = change_info['type']
                
                # Record change event
                self._record_change_event(change_info)
        
        # Check for ongoing adaptations
        if self.recent_changes:
            adaptation_info = self._check_adaptation_progress()
            adaptation_score = adaptation_info['score']
            recovery_time = adaptation_info.get('recovery_time')
        
        return {
            'change_detected': change_detected,
            'change_type': change_type,
            'adaptation_score': adaptation_score,
            'recovery_time': recovery_time,
            'baseline_performance': self.baseline_performance,
            'current_performance': np.mean(list(self.performance_history)[-20:]) if len(self.performance_history) >= 20 else episode_reward
        }
    
    def _detect_environmental_change(self) -> Dict[str, Any]:
        """
        Detect environmental changes using statistical methods.
        
        Returns:
            Dictionary with detection results
        """
        if len(self.performance_history) < self.window_size:
            return {'detected': False}
        
        # Split recent history into comparison windows
        recent_window = list(self.performance_history)[-self.window_size//2:]
        previous_window = list(self.performance_history)[-self.window_size:-self.window_size//2]
        
        # Statistical tests for change detection
        change_detected = False
        change_type = None
        confidence = 0.0
        
        # 1. Mean shift detection
        recent_mean = np.mean(recent_window)
        previous_mean = np.mean(previous_window)
        
        if previous_mean > 0:
            performance_change = (recent_mean - previous_mean) / abs(previous_mean)
        else:
            performance_change = recent_mean - previous_mean
        
        # Significant performance drop
        if performance_change < -self.sensitivity:
            change_detected = True
            change_type = "performance_drop"
            confidence = abs(performance_change)
        
        # 2. Variance change detection
        recent_var = np.var(recent_window)
        previous_var = np.var(previous_window)
        
        if previous_var > 0:
            variance_change = abs(recent_var - previous_var) / previous_var
            if variance_change > self.sensitivity * 2:
                change_detected = True
                change_type = "variance_change" if not change_type else change_type
                confidence = max(confidence, variance_change)
        
        # 3. Distribution shift detection (Kolmogorov-Smirnov test)
        try:
            ks_stat, p_value = stats.ks_2samp(previous_window, recent_window)
            if p_value < 0.05:  # Significant distribution difference
                change_detected = True
                change_type = "distribution_shift" if not change_type else change_type
                confidence = max(confidence, 1 - p_value)
        except:
            pass  # Skip if statistical test fails
        
        # 4. Difficulty correlation check
        if len(self.difficulty_history) >= self.window_size:
            recent_difficulty = list(self.difficulty_history)[-self.window_size//2:]
            previous_difficulty = list(self.difficulty_history)[-self.window_size:-self.window_size//2]
            
            difficulty_change = np.mean(recent_difficulty) - np.mean(previous_difficulty)
            
            # If difficulty increased significantly, expect performance drop
            if difficulty_change > 0.1 and performance_change < -self.sensitivity:
                change_detected = True
                change_type = "difficulty_induced"
                confidence = max(confidence, difficulty_change + abs(performance_change))
        
        return {
            'detected': change_detected,
            'type': change_type,
            'confidence': confidence,
            'performance_change': performance_change,
            'episode': len(self.performance_history)
        }
    
    def _record_change_event(self, change_info: Dict[str, Any]):
        """Record a detected environmental change."""
        change_event = {
            'episode': len(self.performance_history),
            'type': change_info['type'],
            'confidence': change_info['confidence'],
            'performance_change': change_info['performance_change'],
            'pre_change_performance': np.mean(list(self.performance_history)[-self.window_size:-self.window_size//2]),
            'post_change_performance': np.mean(list(self.performance_history)[-self.window_size//2:]),
            'adaptation_start_episode': len(self.performance_history),
            'adapted': False,
            'adaptation_time': None
        }
        
        self.recent_changes.append(change_event)
        self.total_changes_detected += 1
        
        # Keep only recent changes for adaptation tracking
        if len(self.recent_changes) > 5:
            self.recent_changes.pop(0)
    
    def _check_adaptation_progress(self) -> Dict[str, Any]:
        """
        Check progress of ongoing adaptations.
        
        Returns:
            Dictionary with adaptation progress information
        """
        if not self.recent_changes:
            return {'score': 1.0}
        
        adaptation_scores = []
        
        for change_event in self.recent_changes:
            if change_event['adapted']:
                adaptation_scores.append(1.0)
                continue
            
            # Calculate episodes since change
            episodes_since_change = len(self.performance_history) - change_event['adaptation_start_episode']
            
            if episodes_since_change < 10:
                # Too early to judge adaptation
                adaptation_scores.append(0.5)
                continue
            
            # Get performance after change
            recent_performance = np.mean(list(self.performance_history)[-20:])
            pre_change_performance = change_event['pre_change_performance']
            
            # Calculate adaptation score
            if pre_change_performance > 0:
                recovery_ratio = recent_performance / pre_change_performance
                adaptation_score = min(1.0, max(0.0, recovery_ratio))
            else:
                adaptation_score = 0.5 if recent_performance > change_event['post_change_performance'] else 0.0
            
            adaptation_scores.append(adaptation_score)
            
            # Check if adaptation is complete
            if adaptation_score >= self.adaptation_threshold and episodes_since_change >= 20:
                change_event['adapted'] = True
                change_event['adaptation_time'] = episodes_since_change
                self.successful_adaptations += 1
                self.adaptation_times.append(episodes_since_change)
                
                # Move to completed adaptations
                self.adaptation_events.append(change_event.copy())
        
        # Remove completed adaptations from recent changes
        self.recent_changes = [change for change in self.recent_changes if not change['adapted']]
        
        # Calculate overall adaptation score
        overall_score = np.mean(adaptation_scores) if adaptation_scores else 1.0
        
        # Calculate recovery time for most recent change
        recovery_time = None
        if self.recent_changes:
            most_recent = self.recent_changes[-1]
            recovery_time = len(self.performance_history) - most_recent['adaptation_start_episode']
        
        return {
            'score': overall_score,
            'recovery_time': recovery_time,
            'active_adaptations': len(self.recent_changes)
        }
    
    def predict_upcoming_changes(self, lookback: int = 50) -> Dict[str, Any]:
        """
        Predict potential upcoming environmental changes based on patterns.
        
        Args:
            lookback: Number of recent episodes to analyze
        
        Returns:
            Prediction information
        """
        if len(self.difficulty_history) < lookback:
            return {'prediction': 'insufficient_data'}
        
        recent_difficulty = list(self.difficulty_history)[-lookback:]
        recent_performance = list(self.performance_history)[-lookback:]
        
        # Analyze difficulty trends
        difficulty_trend = np.polyfit(range(len(recent_difficulty)), recent_difficulty, 1)[0]
        
        # Analyze performance-difficulty correlation
        if len(recent_difficulty) == len(recent_performance):
            correlation = np.corrcoef(recent_difficulty, recent_performance)[0, 1]
        else:
            correlation = 0.0
        
        # Predict change likelihood
        change_likelihood = 0.0
        
        # Rising difficulty trend
        if difficulty_trend > 0.01:
            change_likelihood += 0.3
        
        # Strong negative correlation (performance drops with difficulty)
        if correlation < -0.5:
            change_likelihood += 0.4
        
        # Recent performance volatility
        performance_volatility = np.std(recent_performance[-20:]) if len(recent_performance) >= 20 else 0
        if performance_volatility > np.std(recent_performance) * 1.5:
            change_likelihood += 0.3
        
        change_likelihood = min(1.0, change_likelihood)
        
        prediction_type = "no_change"
        if change_likelihood > 0.7:
            prediction_type = "likely_change"
        elif change_likelihood > 0.4:
            prediction_type = "possible_change"
        
        return {
            'prediction': prediction_type,
            'likelihood': change_likelihood,
            'difficulty_trend': difficulty_trend,
            'performance_correlation': correlation,
            'performance_volatility': performance_volatility
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        adaptation_rate = self.successful_adaptations / max(1, self.total_changes_detected)
        avg_adaptation_time = np.mean(self.adaptation_times) if self.adaptation_times else 0.0
        
        return {
            'total_episodes': len(self.performance_history),
            'total_changes_detected': self.total_changes_detected,
            'successful_adaptations': self.successful_adaptations,
            'adaptation_rate': adaptation_rate,
            'average_adaptation_time': avg_adaptation_time,
            'current_adaptations': len(self.recent_changes),
            'baseline_performance': self.baseline_performance,
            'current_performance': np.mean(list(self.performance_history)[-20:]) if len(self.performance_history) >= 20 else 0.0
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current detector state for saving/loading."""
        return {
            'performance_history': list(self.performance_history),
            'difficulty_history': list(self.difficulty_history),
            'baseline_performance': self.baseline_performance,
            'recent_changes': self.recent_changes.copy(),
            'adaptation_events': self.adaptation_events.copy(),
            'total_changes_detected': self.total_changes_detected,
            'successful_adaptations': self.successful_adaptations,
            'adaptation_times': self.adaptation_times.copy()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load detector state from saved data."""
        self.performance_history = deque(state['performance_history'], maxlen=self.window_size * 3)
        self.difficulty_history = deque(state['difficulty_history'], maxlen=self.window_size * 3)
        self.baseline_performance = state['baseline_performance']
        self.recent_changes = state['recent_changes']
        self.adaptation_events = state['adaptation_events']
        self.total_changes_detected = state['total_changes_detected']
        self.successful_adaptations = state['successful_adaptations']
        self.adaptation_times = state['adaptation_times']
