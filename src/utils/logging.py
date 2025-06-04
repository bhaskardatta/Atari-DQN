"""
Logging utilities for the adaptive RL agent.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        format_string: Optional custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        Configured experiment logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    return setup_logger(
        name=f"experiment.{experiment_name}",
        log_file=log_file,
        level=level
    )


class MetricsLogger:
    """Logger for training metrics and performance data."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.training_log = self.log_dir / f"{experiment_name}_training.csv"
        self.adaptation_log = self.log_dir / f"{experiment_name}_adaptation.csv"
        self.performance_log = self.log_dir / f"{experiment_name}_performance.csv"
        
        # Write headers
        self._init_training_log()
        self._init_adaptation_log()
        self._init_performance_log()
    
    def _init_training_log(self):
        """Initialize training log with headers."""
        if not self.training_log.exists():
            with open(self.training_log, 'w') as f:
                f.write("episode,step,reward,epsilon,loss,q_value,duration\n")
    
    def _init_adaptation_log(self):
        """Initialize adaptation log with headers."""
        if not self.adaptation_log.exists():
            with open(self.adaptation_log, 'w') as f:
                f.write("episode,step,change_detected,change_type,adaptation_score,recovery_time\n")
    
    def _init_performance_log(self):
        """Initialize performance log with headers."""
        if not self.performance_log.exists():
            with open(self.performance_log, 'w') as f:
                f.write("episode,avg_reward,std_reward,success_rate,difficulty_level\n")
    
    def log_training_step(
        self,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        loss: float,
        q_value: float,
        duration: float
    ):
        """Log training step data."""
        with open(self.training_log, 'a') as f:
            f.write(f"{episode},{step},{reward},{epsilon:.4f},{loss:.4f},{q_value:.4f},{duration:.2f}\n")
    
    def log_adaptation_event(
        self,
        episode: int,
        step: int,
        change_detected: bool,
        change_type: str,
        adaptation_score: float,
        recovery_time: int
    ):
        """Log adaptation event data."""
        with open(self.adaptation_log, 'a') as f:
            f.write(f"{episode},{step},{change_detected},{change_type},{adaptation_score:.4f},{recovery_time}\n")
    
    def log_performance_evaluation(
        self,
        episode: int,
        avg_reward: float,
        std_reward: float,
        success_rate: float,
        difficulty_level: float
    ):
        """Log performance evaluation data."""
        with open(self.performance_log, 'a') as f:
            f.write(f"{episode},{avg_reward:.2f},{std_reward:.2f},{success_rate:.4f},{difficulty_level:.2f}\n")


class TensorBoardLogger:
    """Wrapper for TensorBoard logging."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Logging disabled.")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled:
            self.writer.close()
