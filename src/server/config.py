"""
MSR Server Configuration

Configuration settings for the MSR REST API server.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Server configuration
SERVER_HOST = os.getenv('MSR_SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('MSR_SERVER_PORT', '5000'))
DEBUG_MODE = os.getenv('MSR_DEBUG', 'False').lower() in ('true', '1', 't')

# Security configuration
SECRET_KEY = os.getenv('MSR_SECRET_KEY', 'default-dev-key-change-in-production')
ENABLE_CORS = os.getenv('MSR_ENABLE_CORS', 'False').lower() in ('true', '1', 't')
CORS_ORIGINS = os.getenv('MSR_CORS_ORIGINS', '*').split(',')
API_KEY_REQUIRED = os.getenv('MSR_API_KEY_REQUIRED', 'False').lower() in ('true', '1', 't')
API_KEY = os.getenv('MSR_API_KEY', None)

# Logging configuration
LOG_LEVEL = os.getenv('MSR_LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('MSR_LOG_FILE', None)
LOG_FORMAT = os.getenv('MSR_LOG_FORMAT', 'json' if LOG_FILE else 'text')

# Agent configuration
DEFAULT_MODEL = os.getenv('MSR_DEFAULT_MODEL', None)
DEFAULT_TEMPERATURE = float(os.getenv('MSR_DEFAULT_TEMPERATURE', '0.7'))
MAX_STEPS = int(os.getenv('MSR_MAX_STEPS', '10'))
SESSION_TIMEOUT = int(os.getenv('MSR_SESSION_TIMEOUT', '3600'))  # 1 hour
REQUIRE_STEP_APPROVAL = os.getenv('MSR_REQUIRE_STEP_APPROVAL', 'True').lower() in ('true', '1', 't')

# Tool permissions
ALLOW_PYTHON_EXECUTION = os.getenv('MSR_ALLOW_PYTHON_EXECUTION', 'False').lower() in ('true', '1', 't')
ALLOW_COMMAND_EXECUTION = os.getenv('MSR_ALLOW_COMMAND_EXECUTION', 'False').lower() in ('true', '1', 't')
ALLOW_FILE_OPERATIONS = os.getenv('MSR_ALLOW_FILE_OPERATIONS', 'True').lower() in ('true', '1', 't')
ALLOW_WEB_SEARCH = os.getenv('MSR_ALLOW_WEB_SEARCH', 'True').lower() in ('true', '1', 't')

# Resource limits
MAX_AGENTS = int(os.getenv('MSR_MAX_AGENTS', '10'))
REQUEST_TIMEOUT = int(os.getenv('MSR_REQUEST_TIMEOUT', '300'))  # 5 minutes

# Create a config dictionary for Flask
def get_flask_config():
    """
    Get configuration dictionary for Flask application.
    
    Returns:
        Dictionary with Flask configuration
    """
    return {
        'SECRET_KEY': SECRET_KEY,
        'DEBUG': DEBUG_MODE,
        'LOG_LEVEL': LOG_LEVEL,
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True,
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max upload size
        'REQUEST_TIMEOUT': REQUEST_TIMEOUT,
    } 