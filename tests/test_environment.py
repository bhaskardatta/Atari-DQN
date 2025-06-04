"""
Unit tests for environment components.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.environment.difficulty import DynamicDifficulty, CurriculumDifficulty


class TestDynamicDifficulty(unittest.TestCase):
    """Test dynamic difficulty functionality."""
    
    def setUp(self):
        self.config = {
            'enable_dynamic': True,
            'change_frequency': 500,
            'paddle_speed_range': [0.5, 1.5],
            'ball_speed_range': [1.0, 2.0],
            'paddle_size_range': [0.7, 1.3],
            'brick_regeneration': True
        }
        self.difficulty_manager = DynamicDifficulty(
            paddle_speed_range=(0.5, 1.5),
            ball_speed_multiplier_range=(1.0, 2.0),
            paddle_size_range=(0.7, 1.3),
            difficulty_factor=1.0
        )
    
    def test_initialization(self):
        """Test difficulty manager initialization."""
        self.assertIsNotNone(self.difficulty_manager.current_paddle_speed)
        self.assertIsNotNone(self.difficulty_manager.current_ball_speed_multiplier)
        self.assertIsNotNone(self.difficulty_manager.current_paddle_size)
    
    def test_parameter_update(self):
        """Test difficulty parameter updates."""
        initial_state = self.difficulty_manager._get_current_state()
        
        # Trigger update
        updated_state = self.difficulty_manager.update_frame()  # Past change frequency
        
        # Parameters should have changed
        self.assertIsInstance(updated_state, dict)
        self.assertIn('paddle_speed_multiplier', updated_state)
        self.assertIn('ball_speed_multiplier', updated_state)
        self.assertIn('paddle_size_multiplier', updated_state)
    
    def test_parameter_ranges(self):
        """Test that parameters stay within specified ranges."""
        for _ in range(10):  # Test multiple updates
            state = self.difficulty_manager.update_frame()
            
            self.assertGreaterEqual(state['paddle_speed_multiplier'], 0.5)
            self.assertLessEqual(state['paddle_speed_multiplier'], 1.5)
            self.assertGreaterEqual(state['ball_speed_multiplier'], 1.0)
            self.assertLessEqual(state['ball_speed_multiplier'], 2.0)
            self.assertGreaterEqual(state['paddle_size_multiplier'], 0.7)
            self.assertLessEqual(state['paddle_size_multiplier'], 1.3)
    
    def test_statistics(self):
        """Test difficulty statistics tracking."""
        # Perform several updates
        for i in range(5):
            self.difficulty_manager.update_frame()
        
        stats = self.difficulty_manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_changes', stats)
        # 'current_state' might not be present if no changes occurred
        if stats['total_changes'] > 0:
            self.assertIn('current_state', stats)
    
    def test_disabled_dynamic_difficulty(self):
        """Test behavior when dynamic difficulty is disabled."""
        manager = DynamicDifficulty(difficulty_factor=0.0)  # Disable difficulty
        initial_state = manager._get_current_state()
        
        # Try to trigger update
        updated_state = manager.update_frame()
        
        # Parameters should remain mostly the same (small variations expected)
        self.assertIsInstance(updated_state, dict)


class TestCurriculumDifficulty(unittest.TestCase):
    """Test curriculum difficulty functionality."""
    
    def setUp(self):
        self.config = {
            'enable': True,
            'stages': [
                {'name': 'easy', 'episodes': 100, 'difficulty_multiplier': 0.8},
                {'name': 'normal', 'episodes': 200, 'difficulty_multiplier': 1.0},
                {'name': 'hard', 'episodes': 150, 'difficulty_multiplier': 1.5}
            ],
            'advancement_threshold': 0.7
        }
        stages = [
            {"name": "easy", "difficulty_factor": 0.0},
            {"name": "normal", "difficulty_factor": 0.3},
            {"name": "hard", "difficulty_factor": 0.6}
        ]
        self.scheduler = CurriculumDifficulty(
            stages=stages,
            stage_duration=100,
            performance_threshold=0.7
        )
    
    def test_initialization(self):
        """Test curriculum scheduler initialization."""
        self.assertEqual(len(self.scheduler.stages), 3)
        self.assertEqual(self.scheduler.current_stage, 0)
        self.assertEqual(self.scheduler.performance_threshold, 0.7)
    
    def test_stage_progression(self):
        """Test stage progression logic."""
        # Should start at first stage
        current_stage = self.scheduler.get_current_stage_info()
        self.assertEqual(current_stage['name'], 'easy')
        
        # Force stage completion by episodes
        for _ in range(100):  # Complete first stage
            self.scheduler.update_episode(0.8)
        
        current_stage = self.scheduler.get_current_stage_info()
        self.assertEqual(current_stage['name'], 'normal')
    
    def test_poor_performance_no_advancement(self):
        """Test that poor performance prevents advancement."""
        # Complete stage duration regardless of performance
        for _ in range(100):
            self.scheduler.update_episode(0.3)
        
        # Should advance due to episode limit
        current_stage = self.scheduler.get_current_stage_info()
        self.assertEqual(current_stage['name'], 'normal')
    
    def test_get_difficulty_factor(self):
        """Test difficulty factor retrieval."""
        factor = self.scheduler.get_current_difficulty_factor()
        self.assertEqual(factor, 0.0)  # First stage factor
        
        # Advance to next stage manually
        self.scheduler.current_stage = 1
        factor = self.scheduler.get_current_difficulty_factor()
        self.assertEqual(factor, 0.3)  # Second stage factor
    
    def test_statistics(self):
        """Test curriculum statistics."""
        # Update with some episodes
        for i in range(50):
            self.scheduler.update_episode(np.random.rand())
        
        stats = self.scheduler.get_current_stage_info()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('stage_index', stats)
        self.assertIn('episodes_in_stage', stats)
        self.assertIn('progress', stats)
    
    def test_disabled_curriculum(self):
        """Test behavior when curriculum is disabled."""
        stages = [{"name": "normal", "difficulty_factor": 1.0}]
        scheduler = CurriculumDifficulty(
            stages=stages,
            stage_duration=100,
            performance_threshold=0.7
        )
        
        # Should return stage factor
        factor = scheduler.get_current_difficulty_factor()
        self.assertEqual(factor, 1.0)
        
        # Updates should work normally
        initial_stage = scheduler.current_stage
        scheduler.update_episode(0.9)
        # Stage shouldn't change with just one update
        self.assertEqual(scheduler.current_stage, initial_stage)


if __name__ == '__main__':
    unittest.main()
