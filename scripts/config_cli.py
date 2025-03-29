#!/usr/bin/env python3
"""
Command-line tool for managing MSR configuration and API keys.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import config_manager, create_env_template

def setup_env_file():
    """Set up a new .env file for the project."""
    # Create template
    template_path = create_env_template()
    
    # Check if .env already exists
    if os.path.exists(".env"):
        print("A .env file already exists.")
        user_input = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if user_input != 'y':
            print("Operation cancelled.")
            return
    
    # Copy template to .env
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()
    
    with open(".env", 'w') as env_file:
        env_file.write(template_content)
    
    print("Created .env file. Please edit it to add your API keys.")
    print(f"File location: {os.path.abspath('.env')}")

def add_api_key(args):
    """Add an API key to the configuration."""
    service = args.service.lower()
    
    # If no key is provided on the command line, prompt for it securely
    api_key = args.key
    if not api_key:
        import getpass
        api_key = getpass.getpass(f"Enter your {service.upper()} API key: ")
    
    # Save the API key
    config_manager.set_api_key(service, api_key)
    
    print(f"{service.upper()} API key saved successfully.")
    print(f"Configuration file: {config_manager.config_file}")

def list_keys(args):
    """List available API keys in the configuration."""
    if "API_KEYS" not in config_manager.config:
        print("No API keys are currently configured.")
        return
    
    api_keys = config_manager.config["API_KEYS"]
    if not api_keys:
        print("No API keys are currently configured.")
        return
    
    print("Configured API keys:")
    print("--------------------")
    for service, key in api_keys.items():
        # Only show first and last 4 characters of the key
        masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
        print(f"{service.upper()}: {masked_key}")

def test_api_key(args):
    """Test that an API key works correctly."""
    service = args.service.lower()
    
    # Get the API key
    api_key = config_manager.get_api_key(service)
    if not api_key:
        print(f"No API key found for {service.upper()}.")
        return
    
    # Perform a simple test based on the service
    if service == "qwen":
        try:
            import transformers
            print("Attempting to connect to Qwen API...")
            # A very simple test - we're not actually calling the API here,
            # just checking that the transformers library is available
            print("Success! The transformers library is available.")
            print("Note: This is a basic check. To fully test the API key,")
            print("run an example from the examples/ directory.")
        except ImportError:
            print("Error: The transformers library is not installed.")
            print("Install it with: pip install transformers")
    elif service == "openai":
        try:
            import openai
            print("Attempting to connect to OpenAI API...")
            try:
                # A simple API call to test the key
                openai.api_key = api_key
                models = openai.Model.list()
                print("Success! Connection to OpenAI API established.")
                print(f"Available models: {len(models['data'])} models")
            except Exception as e:
                print(f"Error connecting to OpenAI API: {e}")
        except ImportError:
            print("Error: The openai library is not installed.")
            print("Install it with: pip install openai")
    else:
        print(f"Testing for {service} is not implemented yet.")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="MSR Configuration Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up initial configuration files")
    
    # Add API key command
    add_key_parser = subparsers.add_parser("add-key", help="Add an API key to the configuration")
    add_key_parser.add_argument("service", help="Service name (e.g., qwen, openai)")
    add_key_parser.add_argument("--key", help="API key (if not provided, will prompt securely)")
    
    # List keys command
    list_parser = subparsers.add_parser("list", help="List configured API keys")
    
    # Test API key command
    test_parser = subparsers.add_parser("test", help="Test an API key")
    test_parser.add_argument("service", help="Service to test (e.g., qwen, openai)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "setup":
        setup_env_file()
    elif args.command == "add-key":
        add_api_key(args)
    elif args.command == "list":
        list_keys(args)
    elif args.command == "test":
        test_api_key(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 