"""
Configuration utilities for the MSR framework.
Handles loading of environment variables, API keys, and other configuration.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Default locations for config files
DEFAULT_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".msr")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.json")
DEFAULT_ENV_FILE = ".env"

class ConfigManager:
    """Manager for handling configuration and API keys."""
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the JSON configuration file (default: ~/.msr/config.json)
            env_file: Path to the .env file for environment variables (default: .env)
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.env_file = env_file or DEFAULT_ENV_FILE
        self.config = {}
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Load configuration
        self._load_config()
        self._load_env()
    
    def _load_config(self) -> None:
        """Load configuration from the JSON config file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file {self.config_file}: {e}")
    
    def _load_env(self) -> None:
        """Load environment variables from .env file if it exists."""
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Checks in this order:
        1. Environment variables
        2. JSON config file
        3. Default value
        
        Args:
            key: The configuration key to look up
            default: Default value if the key is not found
            
        Returns:
            The configuration value or default
        """
        # First check environment variables (highest priority)
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        
        # Then check config file
        config_value = self.config.get(key)
        if config_value is not None:
            return config_value
        
        # Finally return default
        return default
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key to set
            value: The value to set
            save: Whether to save the change to the config file
        """
        self.config[key] = value
        
        if save:
            self.save()
    
    def save(self) -> None:
        """Save the current configuration to the config file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {self.config_file}: {e}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service.
        
        Args:
            service: The service name (e.g., 'qwen', 'openai', 'openrouter')
            
        Returns:
            The API key or None if not found
        """
        # Try service-specific environment variable (e.g., QWEN_API_KEY)
        env_key = f"{service.upper()}_API_KEY"
        api_key = os.environ.get(env_key)
        
        if api_key:
            return api_key
        
        # Try generic API_KEYS dictionary in config
        api_keys = self.config.get("API_KEYS", {})
        return api_keys.get(service)
    
    def set_api_key(self, service: str, api_key: str, save: bool = True) -> None:
        """
        Set an API key for a specific service.
        
        Args:
            service: The service name (e.g., 'qwen', 'openai', 'openrouter')
            api_key: The API key to store
            save: Whether to save to the config file
        """
        # Initialize API_KEYS dict if it doesn't exist
        if "API_KEYS" not in self.config:
            self.config["API_KEYS"] = {}
        
        # Set the API key
        self.config["API_KEYS"][service] = api_key
        
        if save:
            self.save()


# Create a default instance for easy import
config_manager = ConfigManager()


def create_env_template() -> str:
    """
    Create a template .env file with placeholders for API keys.
    
    Returns:
        Path to the created template file
    """
    env_template = """# MSR Framework Environment Variables
# Copy this file to .env and fill in your API keys

# API Keys for different models
QWEN_API_KEY="your-qwen-api-key-here"
OPENAI_API_KEY="your-openai-api-key-here"
OPENROUTER_API_KEY="your-openrouter-api-key-here"

# Model configuration
DEFAULT_MODEL="qwen/qwen2.5-32b-instruct"  # Options: qwen/qwen2.5-32b-instruct, anthropic/claude-3-opus-20240229, etc.
"""
    
    template_path = ".env.template"
    
    with open(template_path, 'w') as f:
        f.write(env_template)
    
    return template_path 