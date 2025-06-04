"""
Configuration management for the adaptive RL agent.
"""

import yaml
from typing import Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    input_shape: tuple = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    hidden_dims: list = None
    activation: str = 'relu'
    dropout: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: float = 0.995
    target_update: int = 1000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 512]


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_episodes: int = 10000
    max_steps_per_episode: int = 10000
    save_frequency: int = 1000
    eval_frequency: int = 500
    eval_episodes: int = 10
    buffer_size: int = 100000
    min_buffer_size: int = 10000
    device: str = "cuda"
    seed: int = 42
    

@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_name: str = "BreakoutNoFrameskip-v4"
    frame_stack: int = 4
    frame_skip: int = 4
    frame_size: list = None
    grayscale: bool = True
    max_no_ops: int = 30
    render_mode: str = None
    
    def __post_init__(self):
        if self.frame_size is None:
            self.frame_size = [84, 84]
    

@dataclass
class DifficultyConfig:
    """Dynamic difficulty configuration."""
    enable_dynamic: bool = True
    change_frequency: int = 100
    paddle_speed_range: list = None
    ball_speed_range: list = None
    paddle_size_range: list = None
    brick_regeneration: bool = False
    
    def __post_init__(self):
        if self.paddle_speed_range is None:
            self.paddle_speed_range = [0.5, 1.5]
        if self.ball_speed_range is None:
            self.ball_speed_range = [1.0, 2.0]
        if self.paddle_size_range is None:
            self.paddle_size_range = [0.7, 1.3]
    

@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enable: bool = True
    stages: list = None
    advancement_threshold: float = 0.6
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                {"name": "normal", "episodes": 100, "difficulty_multiplier": 1.0},
                {"name": "moderate", "episodes": 200, "difficulty_multiplier": 1.1},
                {"name": "hard", "episodes": 300, "difficulty_multiplier": 1.2}
            ]


@dataclass
class AgentConfig:
    """Agent configuration."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    target_update_frequency: int = 1000
    double_dqn: bool = True
    dueling_dqn: bool = False
    noisy_networks: bool = False


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration."""
    type: str = 'standard'
    capacity: int = 100000
    batch_size: int = 32
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001


@dataclass
class AdaptationConfig:
    """Adaptation detection configuration."""
    enable: bool = True
    window_size: int = 20
    performance_threshold: float = 0.3
    variance_threshold: float = 2.0
    kl_threshold: float = 0.2


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    save_logs: bool = True
    tensorboard: bool = True
    save_models: bool = True
    save_videos: bool = True
    video_frequency: int = 100


@dataclass
class PathsConfig:
    """Paths configuration."""
    base_dir: str = './results'
    models_dir: str = 'models'
    logs_dir: str = 'logs'
    videos_dir: str = 'videos'
    plots_dir: str = 'plots'


@dataclass
class Config:
    """Main configuration class."""
    network: NetworkConfig = None
    training: TrainingConfig = None
    environment: EnvironmentConfig = None
    difficulty: DifficultyConfig = None
    curriculum: CurriculumConfig = None
    agent: AgentConfig = None
    replay_buffer: ReplayBufferConfig = None
    adaptation: AdaptationConfig = None
    logging: LoggingConfig = None
    paths: PathsConfig = None
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.environment is None:
            self.environment = EnvironmentConfig()
        if self.difficulty is None:
            self.difficulty = DifficultyConfig()
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBufferConfig()
        if self.adaptation is None:
            self.adaptation = AdaptationConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.paths is None:
            self.paths = PathsConfig()
        if self.difficulty is None:
            self.difficulty = DifficultyConfig()
        if self.curriculum is None:
            self.curriculum = CurriculumConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle field name mapping for replay buffer
        replay_buffer_config = config_dict.get('replay_buffer', {}).copy()
        if 'beta_start' in replay_buffer_config:
            replay_buffer_config['beta'] = replay_buffer_config.pop('beta_start')
        # Remove unsupported fields
        replay_buffer_config = {k: v for k, v in replay_buffer_config.items() 
                               if k in ['type', 'capacity', 'batch_size', 'alpha', 'beta', 'beta_increment']}
        
        return cls(
            network=NetworkConfig(**config_dict.get('network', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            difficulty=DifficultyConfig(**config_dict.get('difficulty', {})),
            curriculum=CurriculumConfig(**config_dict.get('curriculum', {})),
            agent=AgentConfig(**config_dict.get('agent', {})),
            replay_buffer=ReplayBufferConfig(**replay_buffer_config),
            adaptation=AdaptationConfig(**config_dict.get('adaptation', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            paths=PathsConfig(**config_dict.get('paths', {}))
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'network': asdict(self.network),
            'training': asdict(self.training),
            'environment': asdict(self.environment),
            'difficulty': asdict(self.difficulty),
            'curriculum': asdict(self.curriculum)
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'network': asdict(self.network),
            'training': asdict(self.training),
            'environment': asdict(self.environment),
            'difficulty': asdict(self.difficulty),
            'curriculum': asdict(self.curriculum)
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    if Path(config_path).exists():
        return Config.from_yaml(config_path)
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()
