import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config from {config_path}: {e}")
        raise

def load_from_txt(filepath: str) -> str:
    """
    Loads initial prompt details from a specified text file and stores them as a single string.

    Args:
        filepath (str): The path to the text file containing initial prompt details.

    Returns:
        bool: True if loading was successful, False otherwise.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Successfully loaded initial prompt details from: {filepath}")
        return text
    except FileNotFoundError:
        logger.error(f"Initial prompt text file not found at: {filepath}")
        text = None
        return text
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading prompt from {filepath}: {e}", exc_info=True)
        text = None
        return text
    
        
