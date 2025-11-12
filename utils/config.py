"""
Configuration loader for SC Labs

Loads config.yaml from project root. All default values are defined in config.yaml.
This module provides a simple interface to access those values.

Single source of truth: config.yaml
"""

import yaml
from pathlib import Path


class Config:
    """
    Configuration manager that loads config.yaml.
    
    All defaults are defined in config.yaml - no hardcoded fallbacks.
    If config.yaml is missing or invalid, returns empty dict and prints warning.
    """
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"Warning: Config file '{self.config_path}' not found. Using empty config.")
            print("Please create config.yaml in the project root.")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    print(f"Warning: Config file '{self.config_path}' is empty.")
                    return {}
                return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using empty config. Please check config.yaml syntax.")
            return {}
    
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
