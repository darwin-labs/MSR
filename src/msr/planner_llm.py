"""
PlannerLLM module for generating structured research plans.
"""
import os
import sys
import json
import asyncio
import re
import uuid
from typing import Dict, List, Optional, Any, Union, TypedDict
from dataclasses import dataclass, field, asdict

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.llm_service.model_selector import TaskType, get_service_for_task
from src.utils.config import config_manager


@dataclass
class ResearchStep:
    """A single step in a research plan."""
    id: str
    title: str
    description: str
    goal: str  # The specific goal or purpose of this step
    expected_output: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
    """A structured research plan consisting of multiple steps."""
    title: str
    description: str
    objective: str
    steps: List[ResearchStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert the research plan to a JSON string."""
        # Convert dataclasses to dictionaries
        plan_dict = asdict(self)
        return json.dumps(plan_dict, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the research plan to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ResearchPlan':
        """Create a research plan from a JSON string."""
        data = json.loads(json_str)
        
        # Convert step dictionaries to ResearchStep objects
        steps = [
            ResearchStep(
                id=step.get("id", ""),
                title=step.get("title", ""),
                description=step.get("description", ""),
                goal=step.get("goal", "No goal specified"),  # Default value if not found
                expected_output=step.get("expected_output", ""),
                dependencies=step.get("dependencies", [])
            )
            for step in data.get("steps", [])
        ]
        
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            objective=data.get("objective", ""),
            steps=steps,
            metadata=data.get("metadata", {})
        )


class PlannerLLM:
    """
    LLM-based planner for generating research plans with structured steps.
    """
    
    def __init__(
        self,
        service: Optional[OpenRouterService] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_steps: int = 7,  # Maximum number of steps to generate
        **kwargs
    ):
        """
        Initialize the PlannerLLM.
        
        Args:
            service: OpenRouter service instance
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            max_steps: Maximum number of steps to generate (default: 7)
            **kwargs: Additional parameters for generation
        """
        # Create or use provided service
        if service:
            self._service = service
        # Otherwise, create a default service
        else:
            default_model = config_manager.get("PLANNER_MODEL", "google/gemini-2.0-flash-001")
            self._service = create_openrouter_service(model=default_model)
            
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.kwargs = kwargs
        
        # Store the model name to avoid serialization issues
        self.model_name = self._service.default_model
    
    def _create_planning_prompt(
        self, 
        prompt: str, 
        num_steps: int = 5,
        include_metadata: bool = True,
        domain: Optional[str] = None
    ) -> str:
        """
        Create a prompt for generating a research plan.
        
        Args:
            prompt: The main prompt or research question
            num_steps: Number of steps to include in the plan (default: 5)
            include_metadata: Whether to include metadata in the plan (default: True)
            domain: Specific domain for the research (e.g., "AI", "medicine")
            
        Returns:
            Formatted prompt for the LLM
        """
        domain_context = f"in the domain of {domain}" if domain else ""
        
        system_prompt = f"""You are an expert research planner {domain_context}. YOU ARE IN JSON MODE.
Your task is to create a structured research plan in JSON format based on the given prompt or question.

The plan should follow this exact JSON structure:
{{
  "title": "Clear, concise title for the research plan",
  "description": "Brief overview of what this research plan aims to accomplish",
  "objective": "Primary objective or question this research addresses",
  "steps": [
    {{
      "id": "step-1",
      "title": "Brief, descriptive title for the step",
      "description": "Detailed explanation of what this step involves",
      "goal": "Specific purpose or objective of this step (what it aims to achieve)",
      "expected_output": "Clear description of what should result from this step",
      "dependencies": []
    }},
    ...
  ]"""
        
        if include_metadata:
            system_prompt += """,
  "metadata": {
    "difficulty": "beginner|intermediate|advanced",
    "estimated_total_time_hours": 2.5,
    "key_skills_required": ["skill1", "skill2"],
    "recommended_resources": ["resource1", "resource2"]
  }
}"""
        else:
            system_prompt += "\n}"
        
        system_prompt += f"""

Follow these guidelines:
1. IMPORTANT: Analyze the complexity of the research task and adjust the number of steps accordingly:
   - For simple tasks, use fewer steps (3-5 steps)
   - For moderately complex tasks, use a medium number of steps (5-8 steps)
   - For highly complex tasks, use more steps (8-12 steps or more)
   I've suggested {num_steps} steps, but you should adjust based on your assessment of the complexity.
2. Make each step clear, actionable, and self-contained.
3. IMPORTANT: Each step MUST include a specific goal field that clearly states what that step aims to achieve.
4. For dependencies, use the step IDs that the current step depends on.
5. Ensure the steps are in a logical sequence but consider which steps could be performed in parallel.
6. The output MUST be valid, parseable JSON.
7. Do not include any explanations or text outside the JSON structure.

Research Prompt: {prompt}"""

        return system_prompt
    
    async def generate_plan_async(
        self, 
        prompt: str,
        num_steps: int = 5,
        include_metadata: bool = True,
        domain: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResearchPlan:
        """
        Generate a research plan asynchronously.
        
        Args:
            prompt: The main prompt or research question
            num_steps: Suggested number of steps in the plan (default: 5). The LLM will adjust
                       based on task complexity - simpler tasks may have fewer steps, while 
                       more complex tasks will have more steps.
            include_metadata: Whether to include metadata (default: True)
            domain: Specific domain for the research (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            A ResearchPlan object
        """
        # Create the planning prompt
        planning_prompt = self._create_planning_prompt(
            prompt=prompt,
            num_steps=num_steps,
            include_metadata=include_metadata,
            domain=domain
        )
        
        # Get parameters with defaults from adapter
        params = {
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Update with any adapter kwargs
        params.update(self.kwargs)
        
        # Override with any function-specific kwargs
        params.update(kwargs)
        
        # Set format to JSON to help models return well-structured JSON
        # Some providers (e.g., Groq) don't support response_format
        current_model = params.get("model") or self.model_name
        if not current_model or not any(provider in current_model.lower() for provider in ["groq"]):
            params["response_format"] = {"type": "json_object"}
        
        # Create chat messages
        messages = [
            {"role": "system", "content": planning_prompt + "\n\nIMPORTANT: Your response MUST be a valid JSON object."},
            {"role": "user", "content": "Generate a detailed research plan in JSON format. Return ONLY the JSON with no other text."}
        ]
        
        try:
            # Make API call
            response = await self._service.generate_chat_response(
                messages=messages,
                **params
            )
            
            # Extract content from response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                print("Planning LLM response:", content)
                
                # Handle empty response content
                if not content or content.strip() == "":
                    print("Warning: Received empty content from LLM, trying again with explicit JSON instructions")
                    
                    # Try again with more explicit JSON instructions
                    messages = [
                        {"role": "system", "content": planning_prompt + "\n\nVERY IMPORTANT: You MUST return a valid JSON object with the structure specified above."},
                        {"role": "user", "content": "Generate a detailed research plan in JSON format for this task. RESPOND ONLY WITH JSON."}
                    ]
                    
                    # Make second API call with more explicit instructions
                    response = await self._service.generate_chat_response(
                        messages=messages,
                        **params
                    )
                    
                    # Extract content again
                    if "choices" in response and len(response["choices"]) > 0:
                        content = response["choices"][0]["message"]["content"]
                        print("Second attempt LLM response:", content)
                    else:
                        content = ""
                
                # Parse JSON content
                try:
                    # Sometimes the LLM might return markdown-formatted JSON
                    if "```json" in content:
                        # Extract JSON from markdown code block
                        json_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        # Extract from generic code block
                        json_content = content.split("```")[1].strip()
                    else:
                        # Use content as is
                        json_content = content
                    
                    # If content is still empty, create a minimal default plan
                    if not json_content or json_content.strip() == "":
                        print("Warning: Still received empty content from LLM, creating default plan")
                        # Create a minimal default plan based on the prompt
                        default_plan = {
                            "title": f"Research Plan for: {prompt[:50]}{'...' if len(prompt) > 50 else ''}",
                            "description": f"A plan to research the following topic: {prompt}",
                            "objective": prompt,
                            "steps": [
                                {
                                    "id": "step-1",
                                    "title": "Initial Research",
                                    "description": "Conduct initial research on the topic to gather information.",
                                    "goal": "To gather preliminary information on the topic",
                                    "expected_output": "Key findings from initial research",
                                    "dependencies": []
                                },
                                {
                                    "id": "step-2",
                                    "title": "Analyze Findings",
                                    "description": "Analyze the information gathered in the initial research.",
                                    "goal": "To derive insights from the information gathered",
                                    "expected_output": "Analysis of findings with key insights",
                                    "dependencies": ["step-1"]
                                },
                                {
                                    "id": "step-3",
                                    "title": "Prepare Final Report",
                                    "description": "Prepare a final report with the findings and analysis.",
                                    "goal": "To compile findings and insights into a comprehensive report",
                                    "expected_output": "Final report with conclusions and recommendations",
                                    "dependencies": ["step-2"]
                                }
                            ]
                        }
                        return ResearchPlan.from_json(json.dumps(default_plan))
                    
                    # Try to parse as JSON and create ResearchPlan
                    return ResearchPlan.from_json(json_content)
                
                except json.JSONDecodeError as e:
                    # If parsing fails, create a minimal error plan
                    error_step = ResearchStep(
                        id="error-1",
                        title="Error parsing LLM response",
                        description=f"The LLM did not return valid JSON: {str(e)}",
                        goal="To obtain a valid JSON research plan",
                        expected_output="Valid JSON response",
                        dependencies=[]
                    )
                    
                    return ResearchPlan(
                        title="Error: Invalid JSON Response",
                        description="The LLM did not generate a properly formatted research plan.",
                        objective=prompt,
                        steps=[error_step],
                        metadata={"error": str(e), "raw_response": content}
                    )
            
            # Handle case where no valid response was received
            error_step = ResearchStep(
                id="error-1",
                title="Error: No valid response received",
                description="The LLM did not return a valid response.",
                goal="To obtain a valid research plan response",
                expected_output="Valid research plan",
                dependencies=[]
            )
            
            return ResearchPlan(
                title="Error: No Response",
                description="The LLM did not generate a research plan.",
                objective=prompt,
                steps=[error_step],
                metadata={"raw_response": str(response)}
            )
            
        except Exception as e:
            # Handle any exceptions
            error_step = ResearchStep(
                id="error-1",
                title="Error generating research plan",
                description=f"An error occurred: {str(e)}",
                goal="To resolve the error and generate a valid research plan",
                expected_output="Valid research plan",
                dependencies=[]
            )
            
            return ResearchPlan(
                title="Error: Generation Failed",
                description="Failed to generate research plan due to an error.",
                objective=prompt,
                steps=[error_step],
                metadata={"error": str(e)}
            )
    
    def generate_plan(
        self, 
        prompt: str,
        num_steps: int = 5,
        include_metadata: bool = True,
        domain: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResearchPlan:
        """
        Generate a research plan (synchronous wrapper).
        
        Args:
            prompt: The main prompt or research question
            num_steps: Suggested number of steps in the plan (default: 5). The LLM will adjust
                       based on task complexity - simpler tasks may have fewer steps, while 
                       more complex tasks will have more steps.
            include_metadata: Whether to include metadata (default: True)
            domain: Specific domain for the research (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            A ResearchPlan object
        """
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate_plan_async(
                    prompt=prompt,
                    num_steps=num_steps,
                    include_metadata=include_metadata,
                    domain=domain,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            )
        finally:
            loop.close()


# Convenience function to create a planner
def create_planner(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> PlannerLLM:
    """
    Create a PlannerLLM instance with configuration.
    
    Args:
        api_key: OpenRouter API key (optional)
        model: Model to use (default: from config or claude)
        temperature: Temperature for generation (default: 0.2)
        max_tokens: Max tokens for generation (default: 2048)
        
    Returns:
        Configured PlannerLLM instance
    """
    # Create the service
    service = create_openrouter_service(api_key=api_key, model=model)
    
    # Create and return the planner
    return PlannerLLM(
        service=service,
        temperature=temperature,
        max_tokens=max_tokens
    ) 