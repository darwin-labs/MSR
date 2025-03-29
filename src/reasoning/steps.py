"""
Implementation of structured reasoning steps for the MSR framework.
"""
from typing import Dict, List, Optional, Union, Any, Callable
import json

class ReasoningStep:
    """Base class for a single reasoning step in the MSR pipeline."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a reasoning step.
        
        Args:
            name: The name of this reasoning step
            description: A description of what this step does
        """
        self.name = name
        self.description = description
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this reasoning step.
        
        Args:
            context: The reasoning context containing all previous step outputs
            
        Returns:
            Updated context with this step's output added
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class PlanningStep(ReasoningStep):
    """A reasoning step that plans the approach to solve a problem."""
    
    def __init__(self, model_fn: Callable, **kwargs):
        """
        Initialize a planning step.
        
        Args:
            model_fn: Function that takes a prompt and returns a model output
            **kwargs: Additional parameters
        """
        super().__init__(
            name="Planning",
            description="Plan the approach to solve the problem"
        )
        self.model_fn = model_fn
        self.kwargs = kwargs
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan for solving the problem."""
        problem = context.get("problem", "")
        
        planning_prompt = (
            f"Problem: {problem}\n\n"
            f"Before solving this problem, I will create a step-by-step plan.\n\n"
            f"My detailed plan to solve this problem:"
        )
        
        plan = self.model_fn(planning_prompt, **self.kwargs)
        
        # Update context with the plan
        context["plan"] = plan
        context["reasoning_steps"] = context.get("reasoning_steps", []) + [
            {"step": self.name, "output": plan}
        ]
        
        return context


class DecompositionStep(ReasoningStep):
    """A reasoning step that breaks down a problem into subproblems."""
    
    def __init__(self, model_fn: Callable, **kwargs):
        """
        Initialize a decomposition step.
        
        Args:
            model_fn: Function that takes a prompt and returns a model output
            **kwargs: Additional parameters
        """
        super().__init__(
            name="Decomposition",
            description="Break down the problem into simpler subproblems"
        )
        self.model_fn = model_fn
        self.kwargs = kwargs
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose the problem into subproblems."""
        problem = context.get("problem", "")
        plan = context.get("plan", "No plan available.")
        
        decomposition_prompt = (
            f"Problem: {problem}\n\n"
            f"Plan: {plan}\n\n"
            f"I will now break this problem down into simpler subproblems:\n"
        )
        
        decomposition = self.model_fn(decomposition_prompt, **self.kwargs)
        
        # Extract subproblems in a structured way if possible
        try:
            # Try to parse if the model returned JSON
            if decomposition.strip().startswith("{") and decomposition.strip().endswith("}"):
                subproblems = json.loads(decomposition)
            else:
                # Otherwise just treat as a list of subproblems in text
                subproblems = decomposition
        except:
            subproblems = decomposition
            
        # Update context with decomposition
        context["decomposition"] = decomposition
        context["subproblems"] = subproblems
        context["reasoning_steps"] = context.get("reasoning_steps", []) + [
            {"step": self.name, "output": decomposition}
        ]
        
        return context


class SolutionStep(ReasoningStep):
    """A reasoning step that solves the subproblems and integrates them."""
    
    def __init__(self, model_fn: Callable, **kwargs):
        """
        Initialize a solution step.
        
        Args:
            model_fn: Function that takes a prompt and returns a model output
            **kwargs: Additional parameters
        """
        super().__init__(
            name="Solution",
            description="Solve each subproblem and integrate solutions"
        )
        self.model_fn = model_fn
        self.kwargs = kwargs
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Solve each subproblem and integrate the solutions."""
        problem = context.get("problem", "")
        plan = context.get("plan", "No plan available.")
        decomposition = context.get("decomposition", "No decomposition available.")
        
        solution_prompt = (
            f"Problem: {problem}\n\n"
            f"Plan: {plan}\n\n"
            f"Problem Decomposition: {decomposition}\n\n"
            f"I will now solve each subproblem and integrate the solutions:\n"
        )
        
        solution = self.model_fn(solution_prompt, **self.kwargs)
        
        # Update context with solution
        context["solution"] = solution
        context["reasoning_steps"] = context.get("reasoning_steps", []) + [
            {"step": self.name, "output": solution}
        ]
        
        return context


class VerificationStep(ReasoningStep):
    """A reasoning step that verifies the solution."""
    
    def __init__(self, model_fn: Callable, **kwargs):
        """
        Initialize a verification step.
        
        Args:
            model_fn: Function that takes a prompt and returns a model output
            **kwargs: Additional parameters
        """
        super().__init__(
            name="Verification",
            description="Verify the solution and identify potential errors"
        )
        self.model_fn = model_fn
        self.kwargs = kwargs
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the solution and identify potential errors."""
        problem = context.get("problem", "")
        solution = context.get("solution", "No solution available.")
        
        verification_prompt = (
            f"Problem: {problem}\n\n"
            f"My solution: {solution}\n\n"
            f"I will now verify my solution by checking for errors and ensuring it correctly solves the original problem:\n"
        )
        
        verification = self.model_fn(verification_prompt, **self.kwargs)
        
        # Update context with verification
        context["verification"] = verification
        context["reasoning_steps"] = context.get("reasoning_steps", []) + [
            {"step": self.name, "output": verification}
        ]
        
        return context


class ReasoningPipeline:
    """A pipeline of reasoning steps to solve a problem."""
    
    def __init__(self, steps: List[ReasoningStep] = None):
        """
        Initialize a reasoning pipeline.
        
        Args:
            steps: List of reasoning steps to execute in order
        """
        self.steps = steps or []
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the pipeline."""
        self.steps.append(step)
    
    def execute(self, problem: str) -> Dict[str, Any]:
        """
        Execute the reasoning pipeline on a problem.
        
        Args:
            problem: The problem to solve
            
        Returns:
            The final reasoning context containing all step outputs
        """
        context = {"problem": problem, "reasoning_steps": []}
        
        for step in self.steps:
            context = step.execute(context)
        
        # Add final answer based on the solution and verification
        solution = context.get("solution", "")
        verification = context.get("verification", "")
        
        final_answer = (
            f"Final Answer:\n\n{solution}\n\n"
            f"Verification: {verification}"
        )
        
        context["final_answer"] = final_answer
        
        return context 