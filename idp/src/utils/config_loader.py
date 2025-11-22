"""Configuration loader utility."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .exceptions import ConfigurationError


class ConfigLoader:
    """Singleton configuration loader."""

    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file. If None, uses default.

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If configuration file cannot be loaded
        """
        if self._config is not None:
            return self._config

        if config_path is None:
            # Default configuration path
            config_path = os.path.join(
                Path(__file__).parent.parent.parent,
                'config',
                'config.yaml'
            )

        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            return self._config
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Unexpected error loading configuration: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'data.test_size')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = ConfigLoader()
            >>> config.get('model.xgboost.base_params.random_state')
            42
        """
        if self._config is None:
            self.load()

        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reload(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Reload configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        self._config = None
        return self.load(config_path)


# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    return _config_loader.load(config_path)


def get_config(key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.

    Args:
        key_path: Dot-separated path to configuration value
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return _config_loader.get(key_path, default)


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Reload configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    return _config_loader.reload(config_path)
