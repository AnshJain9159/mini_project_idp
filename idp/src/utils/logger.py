"""Logging utility module."""

import logging
import logging.config
import os
import yaml
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'diabetes_predictor',
    config_path: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure logger.

    Args:
        name: Logger name
        config_path: Path to logging configuration file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    if config_path is None:
        # Default logging configuration path
        config_path = os.path.join(
            Path(__file__).parent.parent.parent,
            'config',
            'logging_config.yaml'
        )

    # Load logging configuration
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logging.config.dictConfig(config)
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.warning(f"Could not load logging config: {e}. Using basic configuration.")
    else:
        # Basic configuration if config file not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Get logger
    logger = logging.getLogger(name)

    # Override log level if specified
    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger for the application
default_logger = setup_logger()
