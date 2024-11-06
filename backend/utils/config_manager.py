# utils/config_manager.py

import os
from typing import Dict, Any, Optional, List
import yaml
from yaml import SafeLoader
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelConfig:
    """Model configuration settings."""
    sequence_length: int = 60
    lstm_units: int = 128
    dropout_rate: float = 0.2
    dense_units: int = 16
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'ema_9', 'bb_upper', 'bb_lower'
            ]

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    loss_function: str = 'binary_crossentropy'
    model_checkpoint_path: str = 'models/checkpoints/best_model.h5'
    use_tensorboard: bool = True
    tensorboard_dir: str = 'logs/tensorboard'

@dataclass
class SignalConfig:
    """Signal generation configuration settings."""
    confidence_threshold: float = 0.85
    risk_reward_min: float = 2.0
    use_ml_predictions: bool = True
    max_workers: int = 4
    time_frames: List[str] = None
    
    def __post_init__(self):
        if self.time_frames is None:
            self.time_frames = ['1min', '5min', '15min', '30min', 'daily']

@dataclass
class APIConfig:
    """API configuration settings."""
    alpha_vantage_key: str = None
    finnhub_key: str = None
    rate_limit: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.rate_limit is None:
            self.rate_limit = {
                'alpha_vantage': {
                    'requests_per_minute': 5,
                    'daily_limit': 500
                },
                'finnhub': {
                    'requests_per_minute': 60,
                    'daily_limit': 1000
                }
            }

class ConfigManager:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or 'config/config.yaml'
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.signal_config = SignalConfig()
        self.api_config = APIConfig()
        
        if os.path.exists(self.config_path):
            self.load_config()
        else:
            self.save_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Update configurations
            self._update_config(self.model_config, config.get('model', {}))
            self._update_config(self.training_config, config.get('training', {}))
            self._update_config(self.signal_config, config.get('signal', {}))
            self._update_config(self.api_config, config.get('api', {}))
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")

    def save_config(self):
        """Save configuration to file."""
        try:
            config = {
                'model': self._to_dict(self.model_config),
                'training': self._to_dict(self.training_config),
                'signal': self._to_dict(self.signal_config),
                'api': self._to_dict(self.api_config),
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Error saving config: {str(e)}")

    @staticmethod
    def _update_config(config_obj: Any, config_dict: Dict):
        """Update configuration object from dictionary."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    @staticmethod
    def _to_dict(config_obj: Any) -> Dict:
        """Convert configuration object to dictionary."""
        return {k: v for k, v in config_obj.__dict__.items() 
                if not k.startswith('_')}

    def update_config(self, section: str, updates: Dict):
        """Update specific configuration section."""
        config_map = {
            'model': self.model_config,
            'training': self.training_config,
            'signal': self.signal_config,
            'api': self.api_config
        }
        
        if section in config_map:
            self._update_config(config_map[section], updates)
            self.save_config()
        else:
            raise ValueError(f"Invalid configuration section: {section}")

    def get_config(self, section: str) -> Dict:
        """Get configuration for specific section."""
        config_map = {
            'model': self.model_config,
            'training': self.training_config,
            'signal': self.signal_config,
            'api': self.api_config
        }
        
        if section in config_map:
            return self._to_dict(config_map[section])
        raise ValueError(f"Invalid configuration section: {section}")
