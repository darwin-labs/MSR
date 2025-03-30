"""
MSR Agent module for orchestrating task execution using planning and step execution.
"""
import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Set, Type
from enum import Enum, auto
from dataclasses import dataclass, field, asdict

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.utils.config import config_manager
from src.msr.planner_llm import PlannerLLM, ResearchPlan, ResearchStep
from src.msr.step_executor import StepExecutor, StepResult


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
        planner_model: Optional[str] = None,
        executor_model: Optional[str] = None,
        planner_temperature: float = 0.2,
        executor_temperature: float = 0.7,
        max_tokens: int = 2048,
        requires_tool_approval: bool = True,
        save_state_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Agent.
        
        Args:
            task: The task to perform
            allowed_tools: Set of allowed tool types
            planner_model: Model for plan generation
            executor_model: Model for step execution
            planner_temperature: Temperature for plan generation
            executor_temperature: Temperature for step execution
            max_tokens: Max tokens for generation
            requires_tool_approval: Whether tools require approval
            save_state_path: Path to save agent state
            **kwargs: Additional parameters
        """
        self.task = task
        
        # Initialize state
        self.state = AgentState(task=task)
        
        # Set up default allowed tools if not provided
        if allowed_tools is None:
            allowed_tools = {ToolType.PYTHON_CODE, ToolType.DATA_ANALYSIS}
        self.allowed_tools = allowed_tools
        
        # Configure tool approval
        self.requires_tool_approval = requires_tool_approval
        
        # Get models from config or parameters
        self.planner_model = planner_model or config_manager.get("PLANNER_MODEL", "anthropic/claude-3-opus-20240229")
        self.executor_model = executor_model or config_manager.get("EXECUTOR_MODEL", "anthropic/claude-3-opus-20240229")
        
        # Set temperatures
        self.planner_temperature = planner_temperature
        self.executor_temperature = executor_temperature
        
        # Set max tokens
        self.max_tokens = max_tokens
        
        # Additional parameters
        self.kwargs = kwargs
        
        # State saving path
        self.save_state_path = save_state_path
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize the planner and executor components."""
        # Create planner
        self.planner = PlannerLLM(
            model=self.planner_model,
            temperature=self.planner_temperature,
            max_tokens=self.max_tokens
        )
        
        # Create executor
        allow_code_execution = ToolType.PYTHON_CODE in self.allowed_tools
        allow_command_execution = ToolType.TERMINAL_COMMAND in self.allowed_tools
        
        self.executor = StepExecutor(
            model=self.executor_model,
            temperature=self.executor_temperature,
            max_tokens=self.max_tokens,
            allow_code_execution=allow_code_execution,
            allow_command_execution=allow_command_execution
        )
    
    def save_state(self):
        """Save the current state of the agent."""
        if self.save_state_path:
            with open(self.save_state_path, 'w') as f:
                f.write(self.state.to_json())
    
    def load_state(self, state_path: Optional[str] = None):
        """
        Load agent state from file.
        
        Args:
            state_path: Path to the state file, defaults to save_state_path
        """
        path = state_path or self.save_state_path
        if not path or not os.path.exists(path):
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
            
            return True
            
        except Exception as e:
            print(f"Error loading state: {str(e)}")
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
        
        # Generate the plan
        self.state.plan = await self.planner.generate_plan_async(
            prompt=prompt,
            num_steps=num_steps,
            domain=domain,
            **kwargs
        )
        
        # Save state after generating plan
        self.save_state()
        
        return self.state.plan
    
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
            raise ValueError("No plan generated. Call generate_plan first.")
        
        # Determine which step to execute
        if step_index is None:
            step_index = self.state.current_step_index
        
        # Check if step index is valid
        if step_index < 0 or step_index >= len(self.state.plan.steps):
            raise ValueError(f"Step index {step_index} out of range (0-{len(self.state.plan.steps)-1})")
        
        # Get the step to execute
        step = self.state.plan.steps[step_index]
        
        # Get previous results for dependencies
        previous_results = [r for r in self.state.step_results if r.step_id in step.dependencies]
        
        # Execute the step
        result = await self.executor.execute_step_async(
            step=step,
            research_plan=self.state.plan,
            previous_results=previous_results if previous_results else None,
            additional_context=additional_context,
            **kwargs
        )
        
        # Store the result
        self.state.step_results.append(result)
        
        # Update current step index if this is the current step
        if step_index == self.state.current_step_index:
            self.state.current_step_index += 1
        
        # Check if all steps are complete
        if self.state.current_step_index >= len(self.state.plan.steps):
            self.state.is_complete = True
        
        # Save state after executing step
        self.save_state()
        
        return result
    
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
            raise ValueError("No plan generated. Call generate_plan first.")
        
        # Reset execution state
        self.state.current_step_index = 0
        self.state.step_results = []
        self.state.is_complete = False
        
        # Execute all steps
        results = await self.executor.execute_plan_async(
            research_plan=self.state.plan,
            additional_context=additional_context,
            **kwargs
        )
        
        # Update state
        self.state.step_results = results
        self.state.current_step_index = len(self.state.plan.steps)
        self.state.is_complete = True
        
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
    
    def run(
        self,
        num_steps: int = 5,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full agent workflow (generate plan and execute all steps).
        
        Args:
            num_steps: Suggested number of steps
            domain: Specific domain for the research
            additional_context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with plan and execution results
        """
        # Generate plan
        plan = self.generate_plan(
            num_steps=num_steps,
            domain=domain,
            additional_context=additional_context,
            **kwargs
        )
        
        # Execute plan
        results = self.execute_plan(
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
                requires_approval=self.requires_tool_approval
            ))
        
        if ToolType.TERMINAL_COMMAND in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.TERMINAL_COMMAND,
                name="execute_command",
                description="Execute a terminal command and return the result",
                requires_approval=self.requires_tool_approval
            ))
            
        if ToolType.WEB_SEARCH in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.WEB_SEARCH,
                name="web_search",
                description="Search the web for information",
                requires_approval=self.requires_tool_approval
            ))
            
        if ToolType.FILE_READ in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_READ,
                name="read_file",
                description="Read a file and return its contents",
                requires_approval=self.requires_tool_approval
            ))
            
        if ToolType.FILE_WRITE in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.FILE_WRITE,
                name="write_file",
                description="Write content to a file",
                requires_approval=self.requires_tool_approval
            ))
            
        if ToolType.DATA_ANALYSIS in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.DATA_ANALYSIS,
                name="analyze_data",
                description="Analyze data using pandas and return insights",
                requires_approval=self.requires_tool_approval
            ))
            
        if ToolType.VISUALIZATION in self.allowed_tools:
            tools.append(Tool(
                type=ToolType.VISUALIZATION,
                name="create_visualization",
                description="Create a visualization of data",
                requires_approval=self.requires_tool_approval
            ))
            
        return tools


# Convenience function to create an agent
def create_agent(
    task: str,
    allowed_tools: Optional[Set[ToolType]] = None,
    planner_model: Optional[str] = None,
    executor_model: Optional[str] = None,
    save_state_path: Optional[str] = None
) -> Agent:
    """
    Create an Agent instance for the given task.
    
    Args:
        task: The task to perform
        allowed_tools: Set of allowed tool types
        planner_model: Model for plan generation
        executor_model: Model for step execution
        save_state_path: Path to save agent state
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        task=task,
        allowed_tools=allowed_tools,
        planner_model=planner_model,
        executor_model=executor_model,
        save_state_path=save_state_path
    ) 