"""
Integration of OpenRouter LLM service with the MSR reasoning framework.
"""
import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.reasoning.steps import ReasoningPipeline, PlanningStep, DecompositionStep, SolutionStep, VerificationStep


class OpenRouterMSRAdapter:
    """
    Adapter for using OpenRouter with the MSR framework.
    
    This class provides integration between the OpenRouter service
    and the MSR reasoning pipeline.
    """
    
    def __init__(
        self, 
        service: Optional[OpenRouterService] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the OpenRouter MSR adapter.
        
        Args:
            service: OpenRouter service instance (created if not provided)
            model: Model to use (default: from config or qwen/qwen2.5-32b-instruct)
            temperature: Default temperature for generation
            max_tokens: Default max tokens for generation
            **kwargs: Additional parameters for generation
        """
        # Create service if not provided
        self.service = service or create_openrouter_service(model=model)
        
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
    
    def create_model_function(self) -> Callable:
        """
        Create a model function to use with reasoning steps.
        
        Returns:
            Function that takes a prompt and returns a model output
        """
        def model_fn(prompt: str, **kwargs) -> str:
            """Function to pass to reasoning steps."""
            # Get parameters with defaults from adapter
            params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Update with any adapter kwargs
            params.update(self.kwargs)
            
            # Override with any function-specific kwargs
            params.update(kwargs)
            
            # Run the generation in a new event loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.service.generate_text(prompt, **params)
                )
            finally:
                loop.close()
        
        return model_fn
    
    def create_reasoning_pipeline(self) -> ReasoningPipeline:
        """
        Create a reasoning pipeline using the OpenRouter service.
        
        Returns:
            Configured ReasoningPipeline
        """
        # Create the model function
        model_fn = self.create_model_function()
        
        # Create the pipeline
        pipeline = ReasoningPipeline()
        
        # Add reasoning steps
        pipeline.add_step(PlanningStep(model_fn))
        pipeline.add_step(DecompositionStep(model_fn))
        pipeline.add_step(SolutionStep(model_fn))
        pipeline.add_step(VerificationStep(model_fn))
        
        return pipeline


# Convenience function to create an MSR pipeline with OpenRouter
def create_msr_pipeline(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> ReasoningPipeline:
    """
    Create an MSR reasoning pipeline using OpenRouter.
    
    Args:
        api_key: OpenRouter API key (optional)
        model: Model to use (optional)
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        
    Returns:
        Configured ReasoningPipeline
    """
    # Create the service
    service = create_openrouter_service(api_key=api_key, model=model)
    
    # Create the adapter
    adapter = OpenRouterMSRAdapter(
        service=service,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create and return the pipeline
    return adapter.create_reasoning_pipeline() 