"""
MSR Agent module for orchestrating task execution using planning and step execution.
"""
import os
import sys
import json
import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Set, Type
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service, create_service_for_task
from src.utils.config import config_manager
from src.msr.planner_llm import PlannerLLM, ResearchPlan, ResearchStep
from src.msr.step_executor import StepExecutor, StepResult, create_step_executor
from src.msr.logger import get_logger, LogEventType


class ToolType(Enum):
    """Enum for different types of tools that the agent can use."""
    PYTHON_CODE = auto()
    TERMINAL_COMMAND = auto()
    WEB_SEARCH = auto()
    FILE_READ = auto()
    FILE_WRITE = auto()
    DATA_ANALYSIS = auto()
    VISUALIZATION = auto()


@dataclass
class Tool:
    """Representation of a tool that the agent can use."""
    type: ToolType
    name: str
    description: str
    function: Optional[Callable] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary."""
        result = {
            "type": self.type.name,
            "name": self.name,
            "description": self.description,
            "requires_approval": self.requires_approval,
        }
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass
class AgentState:
    """The state of the agent during task execution."""
    task: str
    plan: Optional[ResearchPlan] = None
    executed_steps: List[StepResult] = field(default_factory=list)
    current_step_index: int = 0
    is_complete: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return {
            "task": self.task,
            "plan": self.plan.to_dict() if self.plan else None,
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "current_step_index": self.current_step_index,
            "is_complete": self.is_complete
        }
    
    def to_json(self) -> str:
        """Convert the state to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class Agent:
    """
    MSR Agent that orchestrates task execution using planning and step execution.
    """
    
    def __init__(
        self,
        task: str,
        allowed_tools: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        planner_llm: Optional[PlannerLLM] = None,
        step_executor: Optional[StepExecutor] = None,
        service: Optional[OpenRouterService] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context: Optional[str] = None,
        save_state_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the MSR Agent.
        
        Args:
            task: Task description
            allowed_tools: List of tools the agent is allowed to use
            agent_id: Unique identifier for this agent
            planner_llm: PlannerLLM instance
            step_executor: StepExecutor instance
            service: OpenRouter service
            model: Model to use for generation
            temperature: Default temperature
            max_tokens: Default max tokens
            context: Additional context
            save_state_path: Path to save agent state
            **kwargs: Additional parameters
        """
        # Create unique agent_id if not provided
        self.agent_id = agent_id or f"agent-{str(uuid.uuid4())[:8]}"
        
        # Create unique task_id for tracking
        self.task_id = f"task-{str(uuid.uuid4())[:8]}"
        
        # Store task and context
        self.task = task
        self.context = context
        
        # Initialize state
        self.state = AgentState(task=task)
        
        # Initialize the tools allowed for this agent
        self.allowed_tools = allowed_tools or []
        
        # Get service or create from model
        self.service = service
        self.model_name = model
        
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Set path for state saving
        self.save_state_path = save_state_path
        
        # Initialize planner and executor
        self.planner = planner_llm
        self.step_executor = step_executor
        
        # Get logger
        self.logger = get_logger()
        
        # Log initialization
        self._log_initialization()
    
    def _log_initialization(self):
        """Log agent initialization."""
        allowed_tool_names = [t.name for t in self.allowed_tools]
        
        self.logger.log_agent_initialized(
            agent_id=self.agent_id,
            task=self.task,
            allowed_tools=allowed_tool_names,
            context={
                "temperature": self.temperature,
                "state_path": self.save_state_path
            }
        )
        
        # Log task started
        self.logger.log_task_started(
            agent_id=self.agent_id,
            task_id=self.task_id,
            task_description=self.task
        )
        
    def save_state(self):
        """Save the current state of the agent."""
        if self.save_state_path:
            try:
                with open(self.save_state_path, 'w') as f:
                    f.write(self.state.to_json())
                
                # Log state saved
                self.logger.log_state_saved(
                    agent_id=self.agent_id,
                    path=self.save_state_path
                )
            except Exception as e:
                # Log error
                self.logger.error(
                    message=f"Failed to save agent state: {str(e)}",
                    event_type=LogEventType.ERROR,
                    agent_id=self.agent_id,
                    context={"path": self.save_state_path}
                )
    
    def load_state(self, state_path: Optional[str] = None):
        """
        Load agent state from file.
        
        Args:
            state_path: Path to the state file, defaults to save_state_path
        """
        path = state_path or self.save_state_path
        if not path or not os.path.exists(path):
            self.logger.warning(
                message=f"State file not found: {path}",
                agent_id=self.agent_id
            )
            return False
        
        try:
            with open(path, 'r') as f:
                state_data = json.load(f)
            
            # Update task
            self.task = state_data.get("task", self.task)
            
            # Recreate plan if it exists
            if state_data.get("plan"):
                self.state.plan = ResearchPlan.from_json(json.dumps(state_data["plan"]))
            
            # Update step index
            self.state.current_step_index = state_data.get("current_step_index", 0)
            
            # Recreate step results
            step_results = []
            for result_data in state_data.get("executed_steps", []):
                result = StepResult(
                    step_id=result_data.get("step_id", ""),
                    success=result_data.get("success", False),
                    findings=result_data.get("findings", ""),
                    learning=result_data.get("learning", ""),
                    next_steps=result_data.get("next_steps", []),
                    artifacts=result_data.get("artifacts", {}),
                    code_executions=result_data.get("code_executions", []),
                    command_executions=result_data.get("command_executions", []),
                    error=result_data.get("error")
                )
                step_results.append(result)
            self.state.executed_steps = step_results
            
            # Update is_complete
            self.state.is_complete = state_data.get("is_complete", False)
            
            # Log state loaded
            self.logger.log_state_loaded(
                agent_id=self.agent_id,
                path=path,
                success=True
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Error loading state: {str(e)}"
            
            # Log error
            self.logger.log_state_loaded(
                agent_id=self.agent_id,
                path=path,
                success=False,
                error=error_msg
            )
            
            return False
    
    async def generate_plan_async(
        self,
        num_steps: int = 5,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> ResearchPlan:
        """
        Generate a plan for the task asynchronously.
        
        Args:
            num_steps: Suggested number of steps
            domain: Specific domain for the research
            additional_context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Generated research plan
        """
        # Include additional context if provided
        prompt = self.task
        if additional_context:
            prompt = f"{prompt}\n\nContext: {additional_context}"
        
        # Log plan generation started
        self.logger.log_plan_generation_started(
            agent_id=self.agent_id,
            task_id=self.task_id,
            prompt=prompt,
            context={
                "num_steps": num_steps,
                "domain": domain
            }
        )
        
        try:
            # Generate the plan
            self.state.plan = await self.planner.generate_plan_async(
                prompt=prompt,
                num_steps=num_steps,
                domain=domain,
                **kwargs
            )
            
            # Log plan generation completed
            if self.state.plan:
                # Check if the plan has an error step
                error_steps = [step for step in self.state.plan.steps if step.id.startswith("error")]
                
                if error_steps:
                    # Log plan generation error
                    self.logger.log_error(
                        message="Plan generation failed with JSON format error",
                        agent_id=self.agent_id,
                        task_id=self.task_id,
                        context={
                            "error_step_title": error_steps[0].title,
                            "error_description": error_steps[0].description,
                            "metadata": self.state.plan.metadata
                        }
                    )
                    self.logger.warning(
                        message=f"Generated plan contains error steps: {error_steps[0].title}",
                        context={
                            "error_description": error_steps[0].description
                        }
                    )
                else:
                    # Log successful plan generation with more details
                    self.logger.log_plan_generation_completed(
                        agent_id=self.agent_id,
                        task_id=self.task_id,
                        plan_title=self.state.plan.title,
                        num_steps=len(self.state.plan.steps),
                        context={
                            "objective": self.state.plan.objective,
                            "description": self.state.plan.description,
                            "steps": [
                                {
                                    "id": step.id,
                                    "title": step.title,
                                    "goal": step.goal,
                                    "dependencies": step.dependencies
                                }
                                for step in self.state.plan.steps
                            ]
                        }
                    )
                    
                    # Print the plan
                    print(f"\n==== GENERATED RESEARCH PLAN ====")
                    print(f"Title: {self.state.plan.title}")
                    print(f"Objective: {self.state.plan.objective}")
                    print(f"Steps: {len(self.state.plan.steps)}")
                    for step in self.state.plan.steps:
                        print(f"\n  Step {step.id}: {step.title}")
                        print(f"  Goal: {step.goal}")
                        print(f"  Dependencies: {step.dependencies if step.dependencies else 'None'}")
                    print("================================\n")
            
            # Save state after generating plan
            self.save_state()
            
            return self.state.plan
            
        except Exception as e:
            # Log any unexpected exceptions during plan generation
            self.logger.log_error(
                message=f"Unexpected error during plan generation: {str(e)}",
                agent_id=self.agent_id,
                task_id=self.task_id,
                context={
                    "exception_type": type(e).__name__,
                    "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt
                }
            )
            raise
    
    def generate_plan(
        self,
        num_steps: int = 5,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> ResearchPlan:
        """
        Generate a plan for the task (synchronous wrapper).
        
        Args:
            num_steps: Suggested number of steps
            domain: Specific domain for the research
            additional_context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Generated research plan
        """
        # Create a new event loop and run the async function
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate_plan_async(
                    num_steps=num_steps,
                    domain=domain,
                    additional_context=additional_context,
                    **kwargs
                )
            )
        finally:
            loop.close()
    
    async def execute_step_async(
        self,
        step_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        allow_code_execution: bool = False,
        allow_command_execution: bool = False,
        allow_web_search: bool = False,
        allow_file_operations: bool = False,
        allow_data_analysis: bool = False,
        tools: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> Optional[StepResult]:
        """
        Execute a single step by ID asynchronously.
        
        Args:
            step_id: ID of the step to execute
            temperature: Override temperature for LLM
            max_tokens: Override max tokens for LLM
            allow_code_execution: Whether to allow Python code execution
            allow_command_execution: Whether to allow terminal command execution
            allow_web_search: Whether to allow web search
            allow_file_operations: Whether to allow file operations
            allow_data_analysis: Whether to allow data analysis
            tools: Specific tools to enable (overrides individual tool flags)
            additional_context: Additional context to provide
            **kwargs: Additional parameters
            
        Returns:
            Step execution result or None if step not found
        """
        # Make sure we have a plan
        if not self.state.plan:
            self.logger.warning(
                message="Cannot execute step without a plan",
                agent_id=self.agent_id,
                task_id=self.task_id
            )
            return None
        
        # Find the step by ID
        step = None
        for s in self.state.plan.steps:
            if s.id == step_id:
                step = s
                break
        
        if not step:
            self.logger.warning(
                message=f"Step {step_id} not found in plan",
                agent_id=self.agent_id,
                task_id=self.task_id
            )
            return None
        
        # Check if step has already been executed
        for executed_step in self.state.executed_steps:
            if executed_step.step_id == step_id:
                self.logger.info(
                    message=f"Step {step_id} has already been executed, returning cached result",
                    agent_id=self.agent_id,
                    task_id=self.task_id
                )
                return executed_step
        
        # Log step execution started
        self.logger.log_step_execution_started(
            agent_id=self.agent_id,
            task_id=self.task_id,
            step_id=step_id,
            step_title=step.title,
            context={
                "step_goal": step.goal,
                "dependencies": step.dependencies
            }
        )
        
        # Get previous results for dependencies
        previous_results = []
        for dep_id in step.dependencies:
            # Find the result for this dependency
            for result in self.state.executed_steps:
                if result.step_id == dep_id:
                    previous_results.append(result)
                    break
        
        # Check if all dependencies have been executed
        if len(previous_results) < len(step.dependencies):
            missing = [dep for dep in step.dependencies if dep not in [r.step_id for r in previous_results]]
            self.logger.warning(
                message=f"Not all dependencies for step {step_id} have been executed. Missing: {missing}",
                agent_id=self.agent_id,
                task_id=self.task_id
            )
            return None
        
        # Create step executor if not exists
        if self.step_executor is None:
            self.step_executor = create_step_executor(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        # Execute the step
        try:
            # Configure execution parameters
            execution_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # Configure tools
            if tools is not None:
                # If specific tools are provided, use those
                execution_params["tools"] = tools
            else:
                # Otherwise use the individual flags
                execution_params["allow_code_execution"] = allow_code_execution
                execution_params["allow_command_execution"] = allow_command_execution
                execution_params["allow_web_search"] = allow_web_search
                execution_params["allow_file_operations"] = allow_file_operations
                execution_params["allow_data_analysis"] = allow_data_analysis
            
            # Configure which tools to use based on the allowed_tools set in the agent
            if self.allowed_tools:
                if "python" in self.allowed_tools or "code" in self.allowed_tools:
                    execution_params["allow_code_execution"] = True
                
                if "terminal" in self.allowed_tools or "command" in self.allowed_tools or "shell" in self.allowed_tools:
                    execution_params["allow_command_execution"] = True
                
                if "web" in self.allowed_tools or "search" in self.allowed_tools:
                    execution_params["allow_web_search"] = True
                
                if "file" in self.allowed_tools:
                    execution_params["allow_file_operations"] = True
                
                if "data" in self.allowed_tools or "analysis" in self.allowed_tools:
                    execution_params["allow_data_analysis"] = True
                
                # Provide the full list of tools
                execution_params["tools"] = self.allowed_tools
            
            # Execute step
            result = await self.step_executor.execute_step_async(
                step=step,
                research_plan=self.state.plan,
                previous_results=previous_results if previous_results else None,
                additional_context=additional_context,
                agent_id_for_logs=self.agent_id,
                task_id_for_logs=self.task_id,
                **execution_params
            )
            
            # Store the result
            self.state.executed_steps.append(result)
            
            # Save state after each step execution
            self.save_state()
            
            # Log step execution completed
            self.logger.log_step_execution_completed(
                agent_id=self.agent_id,
                task_id=self.task_id,
                step_id=step_id,
                success=result.success,
                context={
                    "findings_length": len(result.findings),
                    "learning": result.learning,
                    "error": result.error
                }
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing step {step_id}: {str(e)}"
            self.logger.log_error(
                message=error_msg,
                agent_id=self.agent_id,
                task_id=self.task_id,
                context={
                    "step_id": step_id,
                    "exception_type": type(e).__name__
                }
            )
            return None
    
    def execute_step(
        self,
        step_id: str,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> Optional[StepResult]:
        """
        Execute a single step by ID (synchronous wrapper).
        
        Args:
            step_id: ID of the step to execute
            additional_context: Additional context for execution
            **kwargs: Additional parameters for execution
            
        Returns:
            Result of executing the step or None if step not found
        """
        # Create a new event loop and run the async function
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_step_async(
                    step_id=step_id,
                    additional_context=additional_context,
                    **kwargs
                )
            )
        finally:
            loop.close()
    
    async def execute_plan_async(
        self,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> List[StepResult]:
        """
        Execute all steps in the plan asynchronously.
        
        Args:
            additional_context: Additional context for execution
            **kwargs: Additional parameters for execution
            
        Returns:
            List of results from executing all steps
        """
        # Check if plan exists
        if not self.state.plan or not self.state.plan.steps:
            error_msg = "No plan generated. Call generate_plan first."
            self.logger.error(
                message=error_msg,
                agent_id=self.agent_id,
                task_id=self.task_id
            )
            raise ValueError(error_msg)
        
        # Reset execution state
        self.state.current_step_index = 0
        self.state.executed_steps = []
        self.state.is_complete = False
        
        self.logger.info(
            message=f"Executing plan: {self.state.plan.title} with {len(self.state.plan.steps)} steps",
            agent_id=self.agent_id,
            task_id=self.task_id
        )
        
        # Execute all steps one by one
        results = []
        for i, step in enumerate(self.state.plan.steps):
            try:
                result = await self.execute_step_async(
                    step_id=step.id,
                    additional_context=additional_context,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                self.logger.error(
                    message=f"Failed to execute step {i}: {str(e)}",
                    agent_id=self.agent_id,
                    task_id=self.task_id
                )
                # Create failure result
                result = StepResult(
                    step_id=step.id,
                    success=False,
                    findings="Step execution failed",
                    learning="Handle errors in step execution",
                    error=str(e)
                )
                results.append(result)
        
        # Update state
        self.state.executed_steps = results
        self.state.current_step_index = len(self.state.plan.steps)
        self.state.is_complete = True
        
        # Log task completion
        success_rate = len([r for r in results if r.success]) / len(results) if results else 0
        self.logger.log_task_completed(
            agent_id=self.agent_id,
            task_id=self.task_id,
            success=success_rate > 0.5,  # Consider success if more than half of steps succeeded
            result_summary={
                "total_steps": len(results),
                "successful_steps": len([r for r in results if r.success]),
                "success_rate": success_rate
            }
        )
        
        # Save state after executing plan
        self.save_state()
        
        return results
    
    def execute_plan(
        self,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> List[StepResult]:
        """
        Execute all steps in the plan (synchronous wrapper).
        
        Args:
            additional_context: Additional context for execution
            **kwargs: Additional parameters for execution
            
        Returns:
            List of results from executing all steps
        """
        # Create a new event loop and run the async function
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_plan_async(
                    additional_context=additional_context,
                    **kwargs
                )
            )
        finally:
            loop.close()
    
    async def run(
        self,
        planner_config: Dict[str, Any] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full agent workflow (generate plan and execute all steps).
        
        Args:
            planner_config: Configuration for the planner including num_steps, domain, etc.
            additional_context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with plan and execution results
        """
        self.logger.info(
            message=f"Starting full workflow for task: {self.task}",
            agent_id=self.agent_id,
            task_id=self.task_id
        )
        
        # Set default planner config if not provided
        if planner_config is None:
            planner_config = {"num_steps": 5}
        
        # Extract planner parameters
        num_steps = planner_config.get("max_steps", 5)
        domain = planner_config.get("domain")
        
        # Generate plan asynchronously
        plan = await self.generate_plan_async(
            num_steps=num_steps,
            domain=domain,
            additional_context=additional_context,
            **kwargs
        )
        
        # Execute plan asynchronously
        results = await self.execute_plan_async(
            additional_context=additional_context,
            **kwargs
        )
        
        # Check execution success
        successful_steps = len([r for r in results if r.success])
        success_rate = successful_steps / len(results) if results else 0
        
        # Gather all learnings and findings
        all_learnings = "\n".join([f"- {r.learning}" for r in results if r.success])
        
        # Prepare result summary
        result = {
            "task": self.task,
            "plan": plan.to_dict(),
            "step_results": [r.to_dict() for r in results],
            "success_rate": success_rate,
            "is_complete": self.state.is_complete,
            "summary": {
                "total_steps": len(results),
                "successful_steps": successful_steps,
                "key_learnings": all_learnings
            }
        }
        
        return result
    
    def get_available_tools(self) -> List[Tool]:
        """
        Get the list of available tools for the agent.
        
        Returns:
            List of available Tool objects
        """
        tools = []
        
        if ToolType.PYTHON_CODE in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.PYTHON_CODE,
                name="execute_python",
                description="Execute Python code and return the result",
                requires_approval=True
            ))
        
        if ToolType.TERMINAL_COMMAND in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.TERMINAL_COMMAND,
                name="execute_command",
                description="Execute a terminal command and return the result",
                requires_approval=True
            ))
            
        if ToolType.WEB_SEARCH in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.WEB_SEARCH,
                name="web_search",
                description="Search the web for information",
                requires_approval=True
            ))
            
        if ToolType.FILE_READ in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_READ,
                name="read_file",
                description="Read a file and return its contents",
                requires_approval=True
            ))
            
        if ToolType.FILE_WRITE in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_WRITE,
                name="write_file",
                description="Write content to a file",
                requires_approval=True
            ))
            
        if ToolType.DATA_ANALYSIS in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.DATA_ANALYSIS,
                name="analyze_data",
                description="Analyze data using pandas and return insights",
                requires_approval=True
            ))
            
        if ToolType.VISUALIZATION in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.VISUALIZATION,
                name="create_visualization",
                description="Create a visualization of data",
                requires_approval=True
            ))
            
        return tools


# Convenience function to create an agent
def create_agent(
    task: str,
    allowed_tools: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    save_state_path: Optional[str] = None,
    require_step_approval: bool = True
) -> Agent:
    """
    Create an agent with the specified parameters.
    
    Args:
        task: The task to perform
        allowed_tools: List of tools the agent is allowed to use
        agent_id: Optional agent ID (UUID will be generated if not provided)
        model: Model to use for the agent (will be automatically selected if not provided)
        temperature: Temperature for generation
        save_state_path: Path to save agent state
        require_step_approval: Whether to require approval for step execution
        
    Returns:
        Configured Agent instance
    """
    # Generate agent ID if not provided
    if not agent_id:
        agent_id = str(uuid.uuid4())
    
    # Automatically select the best LLM service for this task
    if model is None:
        try:
            # Use the model selector to get the best service for this task
            service = create_service_for_task(task)
        except Exception as e:
            # Fall back to default model if selection fails
            print(f"Warning: Task-specific model selection failed: {str(e)}")
            print("Using default model instead.")
            service = create_openrouter_service()
    else:
        # Use specified model
        service = create_openrouter_service(model=model)
    
    # Create agent
    agent = Agent(
        task=task,
        allowed_tools=allowed_tools,
        agent_id=agent_id,
        planner_llm=PlannerLLM(
            model=service,
            temperature=temperature,
            max_tokens=2048
        ),
        step_executor=StepExecutor(
            model=service,
            temperature=temperature,
            max_tokens=2048
        ),
        service=service,
        temperature=temperature,
        max_tokens=2048,
        save_state_path=save_state_path,
        require_step_approval=require_step_approval
    )
    
    return agent 