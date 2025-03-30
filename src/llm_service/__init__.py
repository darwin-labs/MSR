"""
LLM Service module for the MSR framework.
Provides integration with various LLM providers like OpenRouter.
"""

from src.llm_service.openrouter_service import (
    OpenRouterService, 
    create_openrouter_service,
    create_service_for_task
)
from src.llm_service.base_service import LLMService
from src.llm_service.msr_integration import OpenRouterMSRAdapter, create_msr_pipeline
from src.llm_service.model_selector import (
    ModelSelector, 
    TaskType, 
    get_model_selector,
    get_service_for_task
)

__all__ = [
    "OpenRouterService", 
    "LLMService", 
    "create_openrouter_service",
    "create_service_for_task",
    "OpenRouterMSRAdapter",
    "create_msr_pipeline",
    "ModelSelector",
    "TaskType",
    "get_model_selector",
    "get_service_for_task"
] 