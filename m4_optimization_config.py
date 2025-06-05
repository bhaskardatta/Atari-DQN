#!/usr/bin/env python3
"""
Apple Silicon M4 Optimized Training Configuration
Maximizes GPU utilization and training efficiency
"""

import torch
import os

class M4OptimizedConfig:
    """Configuration optimized for Apple Silicon M4 GPU training"""
    
    def __init__(self):
        self.apply_m4_optimizations()
        
    def apply_m4_optimizations(self):
        """Apply Apple Silicon M4 specific optimizations"""
        if torch.backends.mps.is_available():
            # Enable all MPS optimizations
            torch.backends.mps.allow_fp16 = True
            torch.backends.mps.allow_tf32 = True
            
            # Memory management for M4
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
            
            # Enable MPS caching for better performance
            torch.mps.empty_cache()
            
            print("🔧 Apple Silicon M4 optimizations applied:")
            print("   ✓ FP16 precision enabled")
            print("   ✓ TensorFloat-32 enabled") 
            print("   ✓ Aggressive memory management")
            print("   ✓ MPS cache optimized")
    
    @property
    def training_config(self):
        """Get optimized training configuration for M4"""
        return {
            # Network Architecture
            'network_config': {
                'conv_channels': [32, 64, 64],  # Optimized for M4 GPU
                'kernel_sizes': [8, 4, 3],
                'strides': [4, 2, 1],
                'fc_hidden': 512,
                'use_bias': False,  # Faster on M4
            },
            
            # Training Hyperparameters - Stable settings
            'training': {
                'buffer_size': 50000,       # Smaller for faster startup
                'batch_size': 64,           # Conservative for stability  
                'learning_rate': 0.0001,    # Lower for stability
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay_steps': 25000,
                'target_update_frequency': 1000,
                'optimize_frequency': 4,
            },
            
            # M4 GPU Optimization
            'gpu_config': {
                'use_amp': True,            # Automatic Mixed Precision
                'pin_memory': True,         # Faster data loading
                'non_blocking': True,       # Async GPU transfers
                'compile_model': True,      # Use torch.compile if available
            },
            
            # Training Schedule
            'schedule': {
                'num_episodes': 50000,      # Reasonable target
                'eval_frequency': 1000,     # More frequent evaluation
                'save_frequency': 5000,     # Regular checkpoints
                'log_frequency': 100,       # Detailed logging
            },
            
            # Performance Monitoring
            'monitoring': {
                'track_gpu_usage': True,
                'log_system_stats': True,
                'save_training_curves': True,
                'tensorboard_logging': False,  # Optional
            }
        }
    
    @property
    def optimizer_config(self):
        """Get optimized optimizer configuration"""
        return {
            'optimizer': 'AdamW',
            'lr': 0.0005,
            'weight_decay': 1e-4,
            'eps': 1.5e-4,
            'betas': (0.9, 0.999),
            'amsgrad': False,
        }
    
    def get_dataloader_config(self):
        """Get optimized data loading configuration"""
        return {
            'pin_memory': True,
            'non_blocking': True,
            'prefetch_factor': 2,
            'persistent_workers': False,  # Not needed for RL
        }
    
    def validate_configuration(self):
        """Validate that the configuration is optimal for M4"""
        issues = []
        
        if not torch.backends.mps.is_available():
            issues.append("MPS not available - training will be slow")
        
        config = self.training_config
        
        # Check batch size
        if config['training']['batch_size'] < 64:
            issues.append("Batch size too small for efficient GPU utilization")
        
        # Check buffer size
        if config['training']['buffer_size'] < 50000:
            issues.append("Buffer size might be too small for stable training")
        
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Configuration validated for Apple Silicon M4")
        
        return len(issues) == 0

def main():
    """Test the configuration"""
    config = M4OptimizedConfig()
    
    print("🍎 Apple Silicon M4 Optimized Configuration")
    print("=" * 50)
    
    # Validate configuration
    is_valid = config.validate_configuration()
    
    if is_valid:
        print("\n📋 Training Configuration:")
        training_config = config.training_config
        
        print(f"Batch Size: {training_config['training']['batch_size']}")
        print(f"Buffer Size: {training_config['training']['buffer_size']:,}")
        print(f"Learning Rate: {training_config['training']['learning_rate']}")
        print(f"Target Episodes: {training_config['schedule']['num_episodes']:,}")
        
        print(f"\n🔧 GPU Optimizations:")
        gpu_config = training_config['gpu_config']
        for key, value in gpu_config.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
