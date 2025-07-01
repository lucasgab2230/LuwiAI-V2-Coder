"""
Utility functions for configuration management
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    """
    Configuration manager for the SLM project
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self._config = {}
        
    def load(self) -> None:
        """
        Load configuration from file
        """
        if not self.config_path.exists():
            logging.warning(f"Config file not found at {self.config_path}")
            return
            
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value
    
    def save(self) -> None:
        """
        Save configuration to file
        """
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f)

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values
    
    Returns:
        Dictionary of default configuration values
    """
    return {
        "model": {
            "vocab_size": 30000,
            "embedding_dim": 512,
            "hidden_dim": 1024,
            "num_layers": 6,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "data": {
            "max_length": 128,
            "padding_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>"
        }
    }
