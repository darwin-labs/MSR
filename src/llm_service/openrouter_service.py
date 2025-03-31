"""
OpenRouter service implementation for the MSR framework.
Provides integration with the OpenRouter API to access various LLMs.
"""
import os
import json
import aiohttp
from typing import Dict, List, Optional, Any, Union
import logging

from src.llm_service.base_service import LLMService
from src.utils.config import config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenRouter API constants
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_API_BASE}/chat/completions"
OPENROUTER_MODELS_ENDPOINT = f"{OPENROUTER_API_BASE}/models"


class OpenRouterService(LLMService):
    """Service for interacting with the OpenRouter API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        default_model: str = "qwen/qwen2.5-32b-instruct",
        **kwargs
    ):
        """
        Initialize the OpenRouter service.
        
        Args:
            api_key: OpenRouter API key (optional, will use config if not provided)
            default_model: Default model to use (default: qwen/qwen2.5-32b-instruct)
            **kwargs: Additional parameters
        """
        # Get API key from parameters, or config, or environment
        self.api_key = api_key or config_manager.get_api_key("openrouter")
        
        # Set API key in environment if available
        if self.api_key:
            os.environ["OPENROUTER_API_KEY"] = self.api_key
        elif "OPENROUTER_API_KEY" not in os.environ:
            logger.warning(
                "No OpenRouter API key found. API calls will fail. "
                "Set OPENROUTER_API_KEY in your .env file or use "
                "config_manager.set_api_key('openrouter', 'your-key')"
            )
        
        # Default model to use
        self.default_model = default_model
        
        # HTTP session for API calls
        self._session = None
    
    async def _ensure_session(self):
        """Ensure that an HTTP session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def _close_session(self):
        """Close the HTTP session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        # Get API key from environment or stored value
        api_key = os.environ.get("OPENROUTER_API_KEY", self.api_key)
        
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set it using "
                "config_manager.set_api_key('openrouter', 'your-key') "
                "or in the .env file."
            )
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/darwin-labs/MSR",  # Optional but good practice
            "X-Title": "Multi-Step Reasoning Framework"  # Optional but good practice
        }
    
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt using OpenRouter.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            model: Model to use (default: self.default_model)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Convert prompt to chat format
        messages = [{"role": "user", "content": prompt}]
        
        # Get response
        response = await self.generate_chat_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **kwargs
        )
        
        # Extract and return the content
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        
        return ""
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response in a chat context using OpenRouter.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            model: Model to use (default: self.default_model)
            **kwargs: Additional generation parameters
            
        Returns:
            Response object with generated text and metadata
        """
        # Ensure we have a session
        await self._ensure_session()
        
        # Use default model if none specified
        model = model or self.default_model
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            payload[key] = value
        
        try:
            # Make API call
            async with self._session.post(
                OPENROUTER_CHAT_ENDPOINT,
                headers=self._get_headers(),
                json=payload
            ) as response:
                # Check for errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise ValueError(f"OpenRouter API error: {error_text}")
                
                # Parse and return response
                return await response.json()
        
        except Exception as e:
            logger.error(f"Error in OpenRouter API call: {str(e)}")
            raise
    
    async def get_available_models_async(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from OpenRouter.
        
        Returns:
            List of model details
        """
        # Ensure we have a session
        await self._ensure_session()
        
        try:
            # Make API call
            async with self._session.get(
                OPENROUTER_MODELS_ENDPOINT,
                headers=self._get_headers()
            ) as response:
                # Check for errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise ValueError(f"OpenRouter API error: {error_text}")
                
                # Parse and return response
                data = await response.json()
                return data.get("data", [])
        
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            raise
        
        finally:
            # Clean up session
            await self._close_session()
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available model IDs from OpenRouter.
        
        Returns:
            List of model identifiers
        """
        import asyncio
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        try:
            models_data = loop.run_until_complete(self.get_available_models_async())
            # Extract just the model IDs
            return [model.get("id") for model in models_data]
        finally:
            loop.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
    
    async def async_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simplified chat method with better error handling.
        
        Args:
            messages: List of chat messages
            model: Model name to use (default: self.default_model)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with generated content
        """
        # Use default model if none specified
        model_name = model or self.default_model
        
        # Call generate_chat_response with the model name, not the service object
        return await self.generate_chat_response(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @property
    def model(self) -> str:
        """
        Get the current model being used by this service.
        
        Returns:
            Model identifier
        """
        return self.default_model


# Helper function to create an OpenRouterService
def create_openrouter_service(
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> OpenRouterService:
    """
    Create an OpenRouterService with the specified parameters.
    
    Args:
        api_key: OpenRouter API key (optional, will use config if not provided)
        model: Model to use (default: from config or qwen/qwen2.5-32b-instruct)
        
    Returns:
        Configured OpenRouterService instance
    """
    # Get model from config if not provided
    if model is None:
        model = config_manager.get("DEFAULT_MODEL", "qwen/qwen2.5-32b-instruct")
    
    # Create and return service
    return OpenRouterService(api_key=api_key, default_model=model)


def create_service_for_task(
    task_description: str,
    api_key: Optional[str] = None
) -> OpenRouterService:
    """
    Create an OpenRouterService optimized for a specific task.
    This uses the model_selector to determine the best model for the task.
    
    Args:
        task_description: Description of the task
        api_key: OpenRouter API key (optional, will use config if not provided)
        
    Returns:
        OpenRouterService configured with the appropriate model for the task
    """
    # Import here to avoid circular imports
    from src.llm_service.model_selector import get_service_for_task
    
    # Get the appropriate service for the task
    return get_service_for_task(task_description) 