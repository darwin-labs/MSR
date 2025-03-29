"""
Utility functions for constructing prompts in the MSR framework.
"""
from typing import Dict, List, Optional, Union, Any

# Base prompt templates
BASE_REASONING_PROMPT = """
Problem: {problem}

I'll solve this step-by-step using structured reasoning.
"""

MULTI_STEP_REASONING_TEMPLATE = """
Problem: {problem}

Let me solve this by breaking it down into multiple steps.

{reasoning_steps}

Therefore, my final answer is: {final_answer}
"""

# Specialized templates for different reasoning steps
PLANNING_TEMPLATE = """
Problem: {problem}

Let me first create a plan to solve this problem:
"""

DECOMPOSITION_TEMPLATE = """
Problem: {problem}

I'll break this down into smaller, more manageable subproblems:
"""

SOLUTION_TEMPLATE = """
Problem: {problem}

Now I'll solve each subproblem:

{subproblems_solutions}

Integrating these solutions:
"""

VERIFICATION_TEMPLATE = """
Problem: {problem}

My solution: {solution}

Let me verify this solution by checking for errors:
"""

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given variables.
    
    Args:
        template: The prompt template string with {placeholders}
        **kwargs: Variables to substitute into the template
        
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)

def format_reasoning_steps(steps: List[Dict[str, str]]) -> str:
    """
    Format a list of reasoning steps into a cohesive text.
    
    Args:
        steps: List of dictionaries with 'step' and 'output' keys
        
    Returns:
        Formatted string of reasoning steps
    """
    formatted_steps = []
    
    for i, step_info in enumerate(steps):
        step_name = step_info.get("step", f"Step {i+1}")
        step_output = step_info.get("output", "")
        
        formatted_step = f"Step {i+1} ({step_name}):\n{step_output.strip()}"
        formatted_steps.append(formatted_step)
    
    return "\n\n".join(formatted_steps)

def create_multi_step_reasoning_prompt(
    problem: str,
    reasoning_steps: List[Dict[str, str]],
    final_answer: str
) -> str:
    """
    Create a comprehensive multi-step reasoning prompt.
    
    Args:
        problem: The problem statement
        reasoning_steps: List of reasoning steps with their outputs
        final_answer: The final answer to the problem
        
    Returns:
        Formatted multi-step reasoning prompt
    """
    formatted_steps = format_reasoning_steps(reasoning_steps)
    
    return format_prompt(
        MULTI_STEP_REASONING_TEMPLATE,
        problem=problem,
        reasoning_steps=formatted_steps,
        final_answer=final_answer
    ) 