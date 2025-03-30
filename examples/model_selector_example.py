#!/usr/bin/env python3
"""
Model Selector Example

This script demonstrates how to use the model selector functionality in the MSR framework
to automatically select the appropriate LLM for different types of tasks.
"""

import os
import sys
import argparse
import asyncio
from typing import List, Dict, Any

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service import (
    ModelSelector, 
    TaskType, 
    get_model_selector,
    get_service_for_task,
    create_service_for_task
)
from src.utils.config import config_manager


async def test_task(task_description: str, show_details: bool = False):
    """
    Test the model selector with a specific task.
    
    Args:
        task_description: Description of the task
        show_details: Whether to show detailed output
    """
    # Get the model selector
    selector = get_model_selector()
    
    # Detect task type
    task_type = selector.detect_task_type(task_description)
    
    # Get appropriate model
    model = selector.get_model_for_task(task_description)
    
    # Print results
    print(f"\nTask: \"{task_description}\"")
    print(f"Detected Task Type: {task_type.value}")
    print(f"Selected Model: {model}")
    
    if show_details:
        # Get service for this task
        service = selector.get_service_for_task(task_description)
        
        # Generate a short response to demonstrate model usage
        prompt = f"You are specialized in {task_type.value}. Briefly introduce yourself and your specialties."
        response = await service.generate_text(prompt=prompt, max_tokens=200)
        
        # Print response
        print("\nModel Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)


async def run_examples(show_details: bool = False):
    """
    Run examples for different task types.
    
    Args:
        show_details: Whether to show detailed output
    """
    # Define example tasks for each task type
    example_tasks = {
        "coding": "Write a Python function to find the longest common subsequence of two strings.",
        "math": "Calculate the area under the curve y = x^2 from x=0 to x=4 using integration.",
        "science": "Explain the process of photosynthesis in plants and its importance for life on Earth.",
        "research": "Conduct a literature review on the effects of climate change on marine ecosystems.",
        "planning": "Develop a strategic plan for a software product launch, including marketing and sales.",
        "creative": "Write a short story about a robot that develops consciousness.",
        "reasoning": "Analyze the following argument for logical fallacies: 'All birds can fly. Penguins are birds. Therefore, penguins can fly.'",
        "general": "Summarize the current state of artificial intelligence and its applications."
    }
    
    print("=" * 60)
    print("MODEL SELECTOR DEMONSTRATION")
    print("=" * 60)
    
    print("\nTesting task type detection and model selection:")
    
    # Test each example task
    for task_type, task in example_tasks.items():
        await test_task(task, show_details)
    
    # Custom task example
    if show_details:
        print("\n" + "=" * 60)
        print("CUSTOM TASK EXAMPLE")
        print("=" * 60)
        
        custom_task = "Implement a neural network in PyTorch for image classification."
        
        # Create service using convenience function
        service = create_service_for_task(custom_task)
        
        prompt = f"I need to: {custom_task}\nPlease provide a concise outline of how to approach this."
        response = await service.generate_text(prompt=prompt, max_tokens=300)
        
        print(f"\nTask: \"{custom_task}\"")
        print(f"Model: {service.model}")
        print("\nResponse:")
        print("-" * 40)
        print(response)
        print("-" * 40)


async def update_model_map(task_type_str: str, model: str):
    """
    Update the model map with a custom mapping.
    
    Args:
        task_type_str: String representation of the task type
        model: Model to use for the task type
    """
    try:
        # Convert string to enum
        task_type = TaskType(task_type_str.lower())
        
        # Get selector and update map
        selector = get_model_selector()
        selector.update_model_map(task_type, model)
        
        print(f"\nUpdated model map: {task_type.value} -> {model}")
        
        # Test with a task of this type
        if task_type == TaskType.CODING:
            task = "Write a Python function to sort a list of dictionaries by a specific key."
        elif task_type == TaskType.MATH:
            task = "Solve the differential equation dy/dx = y^2 + x."
        elif task_type == TaskType.RESEARCH:
            task = "Research the impact of social media on mental health in adolescents."
        else:
            task = f"This is a {task_type.value} task that requires specialized knowledge."
        
        await test_task(task, True)
        
    except ValueError:
        print(f"Invalid task type: {task_type_str}")
        print(f"Valid task types: {', '.join([t.value for t in TaskType])}")


def main():
    """Parse command line arguments and run the example."""
    parser = argparse.ArgumentParser(description="Demonstrate model selector functionality")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the model selector with examples")
    test_parser.add_argument("--details", action="store_true", help="Show detailed output including model responses")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update the model map")
    update_parser.add_argument("task_type", help=f"Task type to update. Valid types: {', '.join([t.value for t in TaskType])}")
    update_parser.add_argument("model", help="Model to use for the task type")
    
    # Custom task command
    task_parser = subparsers.add_parser("task", help="Test with a custom task")
    task_parser.add_argument("description", help="Task description")
    task_parser.add_argument("--details", action="store_true", help="Show detailed output including model response")
    
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "test":
        asyncio.run(run_examples(args.details))
    elif args.command == "update":
        asyncio.run(update_model_map(args.task_type, args.model))
    elif args.command == "task":
        asyncio.run(test_task(args.description, args.details))
    else:
        # Default to running examples
        asyncio.run(run_examples(False))


if __name__ == "__main__":
    main() 