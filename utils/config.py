"""
Configuration loader for SC Labs
"""

import yaml
from pathlib import Path


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return self._default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return self._default_config()
    
    def _default_config(self):
        """Return default configuration"""
        return {
            'forecasting': {
                'model': 'simple_feedforward',
                'epochs': 10,
                'learning_rate': 0.001,
                'num_layers': 2,
                'hidden_size': 40,
                'verbose': False,
                'show_progress': False
            },
            'inventory': {
                'default_service_level': 0.95,
                'default_lead_time': 2,
                'default_review_period': 1,
                'verbose': True
            },
            'routing': {
                'default_max_payload': 100,
                'default_origin': '08020',
                'algorithm': 'nearest_neighbor',
                'verbose': True
            },
            'data_generation': {
                'default_seed': 42,
                'verbose': True
            },
            'logging': {
                'level': 'INFO',
                'format': 'clean',
                'show_timestamps': False,
                'use_colors': True
            }
        }
    
    def get(self, *keys, default=None):
        """Get configuration value by nested keys"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


# Global config instance
_config = None

def get_config():
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config
