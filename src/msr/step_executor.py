"""
StepExecutor module for executing individual research steps from a research plan.
"""
import os
import sys
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.utils.config import config_manager
from src.msr.planner_llm import ResearchPlan, ResearchStep


@dataclass
class StepResult:
    """Result of executing a research step."""
    step_id: str
    success: bool
    findings: str
    learning: str
    next_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step result to a dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert the step result to a JSON string."""
        return json.dumps(asdict(self), indent=2)


class StepExecutor:
    """
    LLM-based executor that processes individual research steps and captures learnings.
    """
    
    def __init__(
        self,
        service: Optional[OpenRouterService] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,  # Higher temperature for more creative execution
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the StepExecutor.
        
        Args:
            service: OpenRouter service instance (created if not provided)
            model: Model to use (default: from config or anthropic/claude-3-opus)
            temperature: Default temperature for generation (default: 0.7)
            max_tokens: Default max tokens for generation (default: 2048)
            **kwargs: Additional parameters for generation
        """
        # Use Claude as default model for better reasoning
        default_model = model or config_manager.get("EXECUTOR_MODEL", "anthropic/claude-3-opus-20240229")
        
        # Create service if not provided
        self.service = service or create_openrouter_service(model=default_model)
        
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Store execution history
        self.execution_history: List[StepResult] = []
    
    def _create_step_execution_prompt(
        self,
        step: ResearchStep,
        research_plan: ResearchPlan,
        previous_results: Optional[List[StepResult]] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for executing a research step.
        
        Args:
            step: The research step to execute
            research_plan: The overall research plan
            previous_results: Results from previous steps (optional)
            additional_context: Any additional context to provide (optional)
            
        Returns:
            Formatted prompt for the LLM
        """
        # Format dependency information
        dependency_info = ""
        if previous_results and step.dependencies:
            dependency_info = "\n\n## FINDINGS FROM DEPENDENCY STEPS\n"
            for dep_id in step.dependencies:
                for result in previous_results:
                    if result.step_id == dep_id:
                        dependency_info += f"\nFrom step {dep_id}:\n"
                        dependency_info += f"- Learning: {result.learning}\n"
                        dependency_info += f"- Findings: {result.findings}\n"
        
        # Add additional context if provided
        context_info = f"\n\n## ADDITIONAL CONTEXT\n{additional_context}" if additional_context else ""
        
        system_prompt = f"""You are an expert research assistant executing a specific step in a research plan. YOU MUST RETURN OUTPUT IN JSON FORMAT.

## RESEARCH PLAN OVERVIEW
Title: {research_plan.title}
Objective: {research_plan.objective}
Description: {research_plan.description}

## CURRENT STEP TO EXECUTE
Step ID: {step.id}
Title: {step.title}
Goal: {step.goal}
Description: {step.description}
Expected Output: {step.expected_output}{dependency_info}{context_info}

Execute this research step thoroughly. Think step-by-step about how to achieve the goal. 
Your response must be in the following JSON format:

{{
  "findings": "Detailed findings from executing this step. Include all relevant information discovered.",
  "learning": "The key learning or insight gained from this step, summarized concisely.",
  "next_steps": ["Suggestion for next step 1", "Suggestion for next step 2"],
  "artifacts": {{
    "notes": "Any additional notes or information",
    "references": ["Reference 1", "Reference 2"]
  }}
}}

Make sure your findings directly address the goal of this step and provide the expected output.
The 'learning' field should capture the most important insight that should be carried forward to dependent steps.
"""
        return system_prompt
    
    async def execute_step_async(
        self,
        step: ResearchStep,
        research_plan: ResearchPlan,
        previous_results: Optional[List[StepResult]] = None,
        additional_context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a research step asynchronously.
        
        Args:
            step: The research step to execute
            research_plan: The overall research plan
            previous_results: Results from previous steps (optional)
            additional_context: Any additional context (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            A StepResult object containing the execution results
        """
        # Create the execution prompt
        execution_prompt = self._create_step_execution_prompt(
            step=step,
            research_plan=research_plan,
            previous_results=previous_results,
            additional_context=additional_context
        )
        
        # Get parameters with defaults
        params = {
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Update with any instance kwargs
        params.update(self.kwargs)
        
        # Override with any function-specific kwargs
        params.update(kwargs)
        
        # Set format to JSON to help models return well-structured JSON
        params["response_format"] = {"type": "json_object"}
        
        # Create chat messages
        messages = [
            {"role": "system", "content": execution_prompt},
            {"role": "user", "content": f"Execute research step {step.id}: {step.title}"}
        ]
        
        try:
            # Make API call
            response = await self.service.generate_chat_response(
                messages=messages,
                **params
            )
            
            # Extract content from response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                # Parse JSON content
                try:
                    # Handle potential markdown-formatted JSON
                    if "```json" in content:
                        json_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_content = content.split("```")[1].strip()
                    else:
                        json_content = content
                    
                    # Parse the JSON
                    result_data = json.loads(json_content)
                    
                    # Create StepResult
                    step_result = StepResult(
                        step_id=step.id,
                        success=True,
                        findings=result_data.get("findings", ""),
                        learning=result_data.get("learning", ""),
                        next_steps=result_data.get("next_steps", []),
                        artifacts=result_data.get("artifacts", {})
                    )
                    
                    # Add to execution history
                    self.execution_history.append(step_result)
                    
                    return step_result
                
                except json.JSONDecodeError as e:
                    # If parsing fails, create an error result
                    error_result = StepResult(
                        step_id=step.id,
                        success=False,
                        findings="Failed to parse LLM response as JSON",
                        learning="Ensure proper JSON formatting in LLM responses",
                        error=f"JSON decode error: {str(e)}",
                        artifacts={"raw_response": content}
                    )
                    
                    # Add to execution history
                    self.execution_history.append(error_result)
                    
                    return error_result
            
            # Handle case where no valid response was received
            error_result = StepResult(
                step_id=step.id,
                success=False,
                findings="No valid response received from LLM",
                learning="Check API connection and parameters",
                error="No valid response",
                artifacts={"raw_response": str(response)}
            )
            
            # Add to execution history
            self.execution_history.append(error_result)
            
            return error_result
            
        except Exception as e:
            # Handle any exceptions
            error_result = StepResult(
                step_id=step.id,
                success=False,
                findings="Error occurred during step execution",
                learning="Handle exceptions in API calls",
                error=str(e)
            )
            
            # Add to execution history
            self.execution_history.append(error_result)
            
            return error_result
    
    def execute_step(
        self,
        step: ResearchStep,
        research_plan: ResearchPlan,
        previous_results: Optional[List[StepResult]] = None,
        additional_context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a research step (synchronous wrapper).
        
        Args:
            step: The research step to execute
            research_plan: The overall research plan
            previous_results: Results from previous steps (optional)
            additional_context: Any additional context (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            A StepResult object containing the execution results
        """
        import asyncio
        
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_step_async(
                    step=step,
                    research_plan=research_plan,
                    previous_results=previous_results,
                    additional_context=additional_context,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            )
        finally:
            loop.close()
    
    def execute_plan(
        self,
        research_plan: ResearchPlan,
        additional_context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[StepResult]:
        """
        Execute all steps in a research plan, respecting dependencies.
        
        Args:
            research_plan: The research plan to execute
            additional_context: Any additional context (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            List of StepResult objects for all executed steps
        """
        # Reset execution history
        self.execution_history = []
        
        # Track completed steps by ID
        completed_steps: Dict[str, StepResult] = {}
        
        # Process steps in order, respecting dependencies
        for step in research_plan.steps:
            # Check if all dependencies are completed
            if not all(dep_id in completed_steps for dep_id in step.dependencies):
                # Store as failed result if dependencies are not met
                error_result = StepResult(
                    step_id=step.id,
                    success=False,
                    findings="Cannot execute step because dependencies are not satisfied",
                    learning="Ensure proper dependency resolution in research plan",
                    error="Missing dependencies"
                )
                self.execution_history.append(error_result)
                completed_steps[step.id] = error_result
                continue
            
            # Get results from dependencies
            dependency_results = [completed_steps[dep_id] for dep_id in step.dependencies]
            
            # Execute the step
            result = self.execute_step(
                step=step,
                research_plan=research_plan,
                previous_results=dependency_results if dependency_results else None,
                additional_context=additional_context,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Store the result
            completed_steps[step.id] = result
        
        return self.execution_history


# Convenience function to create a step executor
def create_step_executor(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> StepExecutor:
    """
    Create a StepExecutor instance with configuration.
    
    Args:
        api_key: OpenRouter API key (optional)
        model: Model to use (default: from config or claude)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Max tokens for generation (default: 2048)
        
    Returns:
        Configured StepExecutor instance
    """
    # Create the service
    service = create_openrouter_service(api_key=api_key, model=model)
    
    # Create and return the executor
    return StepExecutor(
        service=service,
        temperature=temperature,
        max_tokens=max_tokens
    ) 