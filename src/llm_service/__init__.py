"""
LLM Service module for the MSR framework.
Provides integration with various LLM providers like OpenRouter.
"""

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.llm_service.base_service import LLMService
from src.llm_service.msr_integration import OpenRouterMSRAdapter, create_msr_pipeline

__all__ = [
    "OpenRouterService", 
    "LLMService", 
    "create_openrouter_service",
    "OpenRouterMSRAdapter",
    "create_msr_pipeline"
] 