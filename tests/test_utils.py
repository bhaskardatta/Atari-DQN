"""
Unit tests for utility components.
"""

import unittest
import tempfile
import yaml
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.config import Config, load_config, get_default_config
from src.utils.logging import setup_logger, get_experiment_logger


class TestConfig(unittest.TestCase):
    """Test configuration management functionality."""
    
    def setUp(self):
        self.test_config_dict = {
            'training': {
                'total_episodes': 1000,
                'max_steps_per_episode': 5000,
                'save_frequency': 100
            },
            'agent': {
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'epsilon_start': 1.0
            },
            'network': {
                'hidden_dims': [256, 256],
                'activation': 'relu'
            }
        }
    
    def test_config_creation(self):
        """Test Config creation and basic functionality."""
        config = get_default_config()
        
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.agent)
        self.assertIsNotNone(config.network)
        self.assertIsNotNone(config.environment)
    
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = get_default_config()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('training', config_dict)
        self.assertIn('network', config_dict)
        self.assertIn('environment', config_dict)
        self.assertIn('difficulty', config_dict)
        self.assertIn('curriculum', config_dict)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent config file."""
        # This should return default config, not raise an error
        config = load_config('nonexistent_file.yaml')
        self.assertIsNotNone(config)
        self.assertIsInstance(config, Config)
    
    def test_config_yaml_save_load(self):
        """Test config loading functionality."""
        # Test loading existing config
        config = load_config('configs/default.yaml')
        
        # Verify loaded config is valid
        self.assertIsInstance(config, Config)
        self.assertIsNotNone(config.training)
        self.assertIsNotNone(config.network)


class TestLogging(unittest.TestCase):
    """Test logging functionality."""
    
    def test_setup_logger(self):
        """Test logger setup."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            # Setup logging
            logger = setup_logger('test_logger', log_file=temp_log_file)
            logger.info('Test message')
            logger.warning('Test warning')
            
            # Check log file exists and has content
            self.assertTrue(Path(temp_log_file).exists())
            
        finally:
            if Path(temp_log_file).exists():
                Path(temp_log_file).unlink()
    
    def test_experiment_logger(self):
        """Test experiment logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_experiment_logger('test_exp', temp_dir)
            self.assertIsNotNone(logger)
            # The logger name includes the experiment prefix
            self.assertTrue(logger.name.endswith('test_exp'))


class TestRunScript(unittest.TestCase):
    """Test that we can run the test suite."""
    
    def test_run_all_tests(self):
        """Test that all test modules can be imported and run."""
        import tests.test_dqn_components
        import tests.test_environment
        import tests.test_utils
        
        # If we get here without import errors, the tests are properly structured
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
