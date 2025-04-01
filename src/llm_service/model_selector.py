"""
Model Selector for LLM Service

This module implements a model selection mechanism to choose the appropriate LLM
based on task specialization.
"""

import os
import re
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.utils.config import config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of task types for model specialization."""
    CODING = "coding"
    MATH = "math"
    SCIENCE = "science"
    RESEARCH = "research"
    PLANNING = "planning"
    CREATIVE = "creative"
    REASONING = "reasoning"
    GENERAL = "general"


# Default mapping of task types to models
DEFAULT_MODEL_MAP = {
    TaskType.CODING: "google/gemini-2.0-flash-001",
    TaskType.MATH: "google/gemini-2.0-flash-001",
    TaskType.SCIENCE: "google/gemini-2.0-flash-001",
    TaskType.RESEARCH: "google/gemini-2.0-flash-001",
    TaskType.PLANNING: "google/gemini-2.0-flash-001",
    TaskType.CREATIVE: "google/gemini-2.0-flash-001",
    TaskType.REASONING: "google/gemini-2.0-flash-001",
    TaskType.GENERAL: "google/gemini-2.0-flash-001"
}

# Keywords for identifying task types from task descriptions
TASK_KEYWORDS = {
    TaskType.CODING: [
        "code", "coding", "program", "programming", "develop", "implementation", 
        "function", "algorithm", "debugging", "software", "script", "implementation",
        "python", "javascript", "java", "typescript", "c++", "rust", "golang"
    ],
    TaskType.MATH: [
        "math", "mathematics", "calculation", "compute", "algebra", "calculus", 
        "equation", "statistical", "probability", "numerical", "arithmetic"
    ],
    TaskType.SCIENCE: [
        "science", "scientific", "experiment", "hypothesis", "theory", "physics", 
        "chemistry", "biology", "laboratory", "research", "empirical"
    ],
    TaskType.RESEARCH: [
        "research", "analyze", "investigation", "study", "survey", "literature", 
        "review", "academic", "scholarly", "paper", "publication", "methodology"
    ],
    TaskType.PLANNING: [
        "plan", "planning", "strategy", "roadmap", "schedule", "organize", 
        "outline", "structure", "framework", "steps", "process", "workflow"
    ],
    TaskType.CREATIVE: [
        "creative", "create", "design", "art", "storytelling", "narrative", 
        "fiction", "write", "novel", "poem", "song", "artistic", "imaginative"
    ],
    TaskType.REASONING: [
        "reason", "reasoning", "logic", "logical", "deduce", "inference", 
        "conclusion", "argument", "critique", "evaluate", "analysis", "judge"
    ]
}


class ModelSelector:
    """
    Model selection service that chooses the appropriate LLM based on task specialization.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_map: Optional[Dict[TaskType, str]] = None,
        default_model: str = "google/gemini-2.0-flash-001",
        load_from_config: bool = True
    ):
        """
        Initialize the model selector.
        
        Args:
            api_key: OpenRouter API key (optional, will use config if not provided)
            model_map: Custom mapping of task types to models
            default_model: Default model to use when task type cannot be determined
            load_from_config: Whether to load model map from configuration
        """
        # Get API key
        self.api_key = api_key or config_manager.get_api_key("openrouter")
        
        # Default model when task type cannot be determined
        self.default_model = default_model
        
        # Initialize model map with defaults
        self.model_map = DEFAULT_MODEL_MAP.copy()
        
        # Override with custom map if provided
        if model_map:
            self.model_map.update(model_map)
        
        # Load from config if requested
        if load_from_config:
            self._load_from_config()
        
        # Initialize service cache to avoid recreating services
        self._service_cache = {}
    
    def _load_from_config(self):
        """Load model map from configuration."""
        # Try to load model map from config
        config_map = config_manager.get("MODEL_MAP", {})
        
        # Update model map if found in config
        if config_map:
            try:
                # Process string task types to enum
                for task_str, model in config_map.items():
                    try:
                        task_type = TaskType(task_str.lower())
                        self.model_map[task_type] = model
                    except ValueError:
                        logger.warning(f"Unknown task type in config: {task_str}")
                
                logger.info(f"Loaded model map from config: {len(config_map)} entries")
            except Exception as e:
                logger.error(f"Error loading model map from config: {str(e)}")
    
    def detect_task_type(self, task_description: str) -> TaskType:
        """
        Detect the task type from a task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Detected task type
        """
        # Convert to lowercase for matching
        text = task_description.lower()
        
        # Count keyword matches for each task type
        scores = {task_type: 0 for task_type in TaskType}
        
        # Score each task type based on keyword matches
        for task_type, keywords in TASK_KEYWORDS.items():
            for keyword in keywords:
                # Look for whole word matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, text)
                scores[task_type] += len(matches)
        
        # Find the task type with the highest score
        best_task_type = TaskType.GENERAL
        best_score = 0
        
        for task_type, score in scores.items():
            if score > best_score:
                best_score = score
                best_task_type = task_type
        
        # If no clear match, default to GENERAL
        return best_task_type if best_score > 0 else TaskType.GENERAL
    
    def get_model_for_task(self, task_description: str) -> str:
        """
        Get the appropriate model for a given task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Model ID to use for the task
        """
        # Detect task type
        task_type = self.detect_task_type(task_description)
        
        # Get model for task type
        model = self.model_map.get(task_type, self.default_model)
        
        logger.info(f"Selected model '{model}' for task type '{task_type.value}'")
        return model
    
    def get_service_for_task(self, task_description: str) -> OpenRouterService:
        """
        Get a configured LLM service for a given task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Configured OpenRouterService with the appropriate model
        """
        # Get model for task
        model = self.get_model_for_task(task_description)
        
        # Check if we already have a service for this model
        if model in self._service_cache:
            return self._service_cache[model]
        
        # Create new service
        service = create_openrouter_service(api_key=self.api_key, model=model)
        
        # Cache service
        self._service_cache[model] = service
        
        return service
    
    def update_model_map(self, task_type: TaskType, model: str):
        """
        Update the model map for a specific task type.
        
        Args:
            task_type: Task type to update
            model: Model to use for the task type
        """
        self.model_map[task_type] = model
        logger.info(f"Updated model for {task_type.value} to {model}")


# Global instance for easy access
_default_selector = None

def get_model_selector() -> ModelSelector:
    """
    Get the default model selector instance.
    
    Returns:
        Global ModelSelector instance
    """
    global _default_selector
    
    if _default_selector is None:
        _default_selector = ModelSelector()
    
    return _default_selector

def get_service_for_task(task_description: str) -> OpenRouterService:
    """
    Convenience function to get an LLM service for a specific task.
    
    Args:
        task_description: Description of the task
        
    Returns:
        Configured OpenRouterService with the appropriate model
    """
    selector = get_model_selector()
    return selector.get_service_for_task(task_description) 