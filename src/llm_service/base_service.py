"""
Base class for LLM service providers in the MSR framework.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class LLMService(ABC):
    """Abstract base class for LLM service providers."""
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM service.
        
        Args:
            api_key: API key for the service
            **kwargs: Additional service-specific parameters
        """
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response in a chat context.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Response object with generated text and metadata
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from the service.
        
        Returns:
            List of model identifiers
        """
        pass 