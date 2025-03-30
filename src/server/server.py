#!/usr/bin/env python3
"""
MSR Server Launcher

This script launches the MSR REST API server with the specified configuration.
"""

import os
import sys
import argparse
from typing import Dict, Any

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.server.app import create_app, run_server
from src.server.config import (
    SERVER_HOST, SERVER_PORT, DEBUG_MODE, LOG_LEVEL, LOG_FILE, 
    get_flask_config
)
from src.server.middleware import setup_middleware, cleanup_session
from src.msr.logger import configure_logger, LogLevel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the MSR REST API server")
    
    # Server configuration
    parser.add_argument('--host', type=str, default=SERVER_HOST,
                        help=f'Host to bind the server to (default: {SERVER_HOST})')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help=f'Port to bind the server to (default: {SERVER_PORT})')
    parser.add_argument('--debug', action='store_true', default=DEBUG_MODE,
                        help='Run in debug mode')
    
    # Logging configuration
    parser.add_argument('--log-level', type=str, default=LOG_LEVEL,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help=f'Logging level (default: {LOG_LEVEL})')
    parser.add_argument('--log-file', type=str, default=LOG_FILE,
                        help='File to write logs to')
    parser.add_argument('--log-format', type=str, default=None,
                        choices=['text', 'json'],
                        help='Log format (default: json if log-file is set, otherwise text)')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    
    return parser.parse_args()


def main():
    """Run the MSR REST API server."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_format = args.log_format
    if not log_format:
        log_format = 'json' if args.log_file else 'text'
    
    configure_logger(
        level=LogLevel[args.log_level],
        log_file=args.log_file,
        log_format=log_format
    )
    
    # Load configuration
    config = get_flask_config()
    
    # Override with command line arguments
    config.update({
        'DEBUG': args.debug,
        'LOG_LEVEL': args.log_level,
    })
    
    # Create and configure the app
    app = create_app(config)
    
    # Set up middleware
    setup_middleware(app)
    
    # Set up session cleanup
    cleanup_session(app)
    
    # Run the server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main() 