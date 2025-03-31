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
from src.msr.step_executor import StepExecutor, StepResult
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
    current_step_index: int = 0
    step_results: List[StepResult] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return {
            "task": self.task,
            "plan": self.plan.to_dict() if self.plan else None,
            "current_step_index": self.current_step_index,
            "step_results": [r.to_dict() for r in self.step_results],
            "artifacts": self.artifacts,
            "context": self.context,
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
        allowed_tools: Optional[Set[ToolType]] = None,
        service: OpenRouterService = None,
        agent_id: Optional[str] = None,
        temperature: float = 0.7,
        save_state_path: Optional[str] = None,
        require_step_approval: bool = True,
        **kwargs
    ):
        """
        Initialize the Agent.
        
        Args:
            task: The task to perform
            allowed_tools: Set of allowed tool types
            service: LLM service for the agent
            agent_id: Unique identifier for the agent (generated if not provided)
            temperature: Temperature for generation
            save_state_path: Path to save agent state
            require_step_approval: Whether to require approval for step execution
            **kwargs: Additional parameters
        """
        self.task = task
        
        # Generate unique agent_id if not provided
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        
        # Generate unique task_id for logging
        self.task_id = f"task-{uuid.uuid4().hex[:8]}"
        
        # Get logger
        self.logger = get_logger()
        
        # Initialize state
        self.state = AgentState(task=task)
        
        # Set up default allowed tools if not provided
        if allowed_tools is None:
            allowed_tools = {ToolType.PYTHON_CODE, ToolType.DATA_ANALYSIS}
        self.allowed_tools = allowed_tools
        
        # Configure tool approval
        self.require_step_approval = require_step_approval
        
        # Set temperatures
        self.temperature = temperature
        
        # State saving path
        self.save_state_path = save_state_path
        
        # Additional parameters
        self.kwargs = kwargs
        
        # Initialize components
        self._initialize_components(service)
        
        # Log agent initialization
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
        
    def _initialize_components(self, service: OpenRouterService):
        """Initialize the planner and executor components."""
        # Create planner
        self.planner = PlannerLLM(
            model=service,
            temperature=self.temperature,
            max_tokens=self.kwargs.get("max_tokens", 2048)
        )
        
        # Create executor
        allow_code_execution = ToolType.PYTHON_CODE in self.allowed_tools
        allow_command_execution = ToolType.TERMINAL_COMMAND in self.allowed_tools
        
        self.executor = StepExecutor(
            model=service,
            temperature=self.temperature,
            max_tokens=self.kwargs.get("max_tokens", 2048),
            allow_code_execution=allow_code_execution,
            allow_command_execution=allow_command_execution,
            share_output_with_llm=True  # Always share execution results with the LLM
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
            for result_data in state_data.get("step_results", []):
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
            self.state.step_results = step_results
            
            # Update artifacts and context
            self.state.artifacts = state_data.get("artifacts", {})
            self.state.context = state_data.get("context", {})
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
                self.logger.log_plan_generation_completed(
                    agent_id=self.agent_id,
                    task_id=self.task_id,
                    plan_title=self.state.plan.title,
                    num_steps=len(self.state.plan.steps),
                    context={
                        "objective": self.state.plan.objective,
                        "description": self.state.plan.description
                    }
                )
            
            # Save state after generating plan
            self.save_state()
            
            return self.state.plan
            
        except Exception as e:
            # Log error
            self.logger.error(
                message=f"Plan generation failed: {str(e)}",
                event_type=LogEventType.ERROR,
                agent_id=self.agent_id,
                task_id=self.task_id
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
        step_index: Optional[int] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a single step of the plan asynchronously.
        
        Args:
            step_index: Index of the step to execute (default: current step)
            additional_context: Additional context for execution
            **kwargs: Additional parameters for execution
            
        Returns:
            Result of executing the step
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
        
        # Determine which step to execute
        if step_index is None:
            step_index = self.state.current_step_index
        
        # Check if step index is valid
        if step_index < 0 or step_index >= len(self.state.plan.steps):
            error_msg = f"Step index {step_index} out of range (0-{len(self.state.plan.steps)-1})"
            self.logger.error(
                message=error_msg,
                agent_id=self.agent_id,
                task_id=self.task_id
            )
            raise ValueError(error_msg)
        
        # Get the step to execute
        step = self.state.plan.steps[step_index]
        
        # Log step execution started
        self.logger.log_step_execution_started(
            agent_id=self.agent_id,
            task_id=self.task_id,
            step_id=step.id,
            step_title=step.title,
            context={
                "step_goal": step.goal,
                "step_description": step.description
            }
        )
        
        # Get previous results for dependencies
        previous_results = [r for r in self.state.step_results if r.step_id in step.dependencies]
        
        try:
            # Execute the step
            result = await self.executor.execute_step_async(
                step=step,
                research_plan=self.state.plan,
                previous_results=previous_results if previous_results else None,
                additional_context=additional_context,
                **kwargs
            )
            
            # Log step execution completed
            self.logger.log_step_execution_completed(
                agent_id=self.agent_id,
                task_id=self.task_id,
                step_id=step.id,
                success=result.success,
                findings=result.findings[:200] + "..." if len(result.findings) > 200 else result.findings,
                learning=result.learning,
                context={
                    "next_steps": result.next_steps,
                    "error": result.error
                }
            )
            
            # Log any code executions
            for i, code_exec in enumerate(result.code_executions):
                self.logger.log_tool_execution(
                    agent_id=self.agent_id,
                    task_id=self.task_id,
                    tool_name="execute_python",
                    success=code_exec.get("success", False),
                    input_data=code_exec.get("code", "")[:100] + "..." if len(code_exec.get("code", "")) > 100 else code_exec.get("code", ""),
                    output_data=code_exec.get("output", "")[:100] + "..." if len(code_exec.get("output", "")) > 100 else code_exec.get("output", ""),
                    error=code_exec.get("error"),
                    context={
                        "description": code_exec.get("description", ""),
                        "step_id": step.id
                    }
                )
            
            # Log any command executions
            for i, cmd_exec in enumerate(result.command_executions):
                self.logger.log_tool_execution(
                    agent_id=self.agent_id,
                    task_id=self.task_id,
                    tool_name="execute_command",
                    success=cmd_exec.get("success", False),
                    input_data=cmd_exec.get("command", ""),
                    output_data=cmd_exec.get("stdout", "")[:100] + "..." if len(cmd_exec.get("stdout", "")) > 100 else cmd_exec.get("stdout", ""),
                    error=cmd_exec.get("stderr"),
                    context={
                        "description": cmd_exec.get("description", ""),
                        "exit_code": cmd_exec.get("exit_code"),
                        "step_id": step.id
                    }
                )
            
            # Store the result
            self.state.step_results.append(result)
            
            # Update current step index if this is the current step
            if step_index == self.state.current_step_index:
                self.state.current_step_index += 1
            
            # Check if all steps are complete
            if self.state.current_step_index >= len(self.state.plan.steps):
                self.state.is_complete = True
                
                # Log task completion if all steps are done
                if self.state.is_complete:
                    success_rate = len([r for r in self.state.step_results if r.success]) / len(self.state.step_results)
                    
                    self.logger.log_task_completed(
                        agent_id=self.agent_id,
                        task_id=self.task_id,
                        success=success_rate > 0.5,  # Consider success if more than half of steps succeeded
                        result_summary={
                            "total_steps": len(self.state.step_results),
                            "successful_steps": len([r for r in self.state.step_results if r.success]),
                            "success_rate": success_rate
                        }
                    )
            
            # Save state after executing step
            self.save_state()
            
            return result
            
        except Exception as e:
            # Log error
            self.logger.error(
                message=f"Step execution failed: {str(e)}",
                event_type=LogEventType.ERROR,
                agent_id=self.agent_id,
                task_id=self.task_id,
                context={
                    "step_id": step.id,
                    "step_title": step.title
                }
            )
            raise
    
    def execute_step(
        self,
        step_index: Optional[int] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a single step of the plan (synchronous wrapper).
        
        Args:
            step_index: Index of the step to execute (default: current step)
            additional_context: Additional context for execution
            **kwargs: Additional parameters for execution
            
        Returns:
            Result of executing the step
        """
        # Create a new event loop and run the async function
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute_step_async(
                    step_index=step_index,
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
        self.state.step_results = []
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
                    step_index=i,
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
        self.state.step_results = results
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
                requires_approval=self.require_step_approval
            ))
        
        if ToolType.TERMINAL_COMMAND in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.TERMINAL_COMMAND,
                name="execute_command",
                description="Execute a terminal command and return the result",
                requires_approval=self.require_step_approval
            ))
            
        if ToolType.WEB_SEARCH in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.WEB_SEARCH,
                name="web_search",
                description="Search the web for information",
                requires_approval=self.require_step_approval
            ))
            
        if ToolType.FILE_READ in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_READ,
                name="read_file",
                description="Read a file and return its contents",
                requires_approval=self.require_step_approval
            ))
            
        if ToolType.FILE_WRITE in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_WRITE,
                name="write_file",
                description="Write content to a file",
                requires_approval=self.require_step_approval
            ))
            
        if ToolType.DATA_ANALYSIS in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.DATA_ANALYSIS,
                name="analyze_data",
                description="Analyze data using pandas and return insights",
                requires_approval=self.require_step_approval
            ))
            
        if ToolType.VISUALIZATION in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.VISUALIZATION,
                name="create_visualization",
                description="Create a visualization of data",
                requires_approval=self.require_step_approval
            ))
            
        return tools


# Convenience function to create an agent
def create_agent(
    task: str,
    allowed_tools: Optional[Set[ToolType]] = None,
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
        allowed_tools: Set of allowed tool types
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
        service=service,
        agent_id=agent_id,
        temperature=temperature,
        save_state_path=save_state_path,
        require_step_approval=require_step_approval
    )
    
    return agent 