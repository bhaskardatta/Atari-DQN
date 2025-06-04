#!/usr/bin/env python3
"""
Basic functionality test to verify all core components work.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("\nTesting core imports...")
    
    try:
        from src.dqn.agent import DQNAgent
        from src.environment.breakout_env import make_dynamic_breakout
        from src.training.trainer import AdaptiveTrainer
        from src.utils.config import load_config
        from src.analysis.performance import PerformanceAnalyzer
        print("✓ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from src.utils.config import load_config
        
        config = load_config('configs/quick_test.yaml')
        print("✓ Configuration loaded successfully")
        print(f"  - Training episodes: {config.training.total_episodes}")
        print(f"  - Learning rate: {config.agent.learning_rate}")
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_environment_creation():
    """Test that environment can be created."""
    print("\nTesting environment creation...")
    
    try:
        from src.environment.breakout_env import make_dynamic_breakout
        
        env = make_dynamic_breakout(
            difficulty_factor=0.0,
            frame_stack=4,
            render_mode=None
        )
        print("✓ Environment created successfully")
        print(f"  - Action space: {env.action_space.n}")
        print(f"  - Observation space: {env.observation_space.shape}")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment creation error: {e}")
        return False

def test_agent_creation():
    """Test that agent can be created."""
    print("\nTesting agent creation...")
    
    try:
        from src.dqn.agent import DQNAgent
        from src.utils.config import load_config
        
        config = load_config('configs/quick_test.yaml')
        
        agent = DQNAgent(
            state_shape=(4, 84, 84),
            n_actions=4,
            learning_rate=config.agent.learning_rate,
            device='cpu'  # Use CPU for testing
        )
        print("✓ Agent created successfully")
        return True
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return False

def test_directories():
    """Test that required directories exist or can be created."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'results',
        'results/models',
        'results/logs',
        'results/videos',
        'results/plots'
    ]
    
    try:
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("✓ All required directories available")
        return True
    except Exception as e:
        print(f"❌ Directory error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_environment_creation,
        test_agent_creation,
        test_directories
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All basic functionality tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
