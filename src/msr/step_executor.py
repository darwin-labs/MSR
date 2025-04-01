"""
StepExecutor module for executing individual research steps from a research plan.
"""
import os
import sys
import json
import subprocess
import tempfile
import re
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import OpenRouterService, create_openrouter_service
from src.utils.config import config_manager
from src.msr.planner_llm import ResearchPlan, ResearchStep
from src.msr.logger import get_logger, LogEventType


@dataclass
class CodeExecutionResult:
    """Result of executing code."""
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class CommandExecutionResult:
    """Result of executing a terminal command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class StepResult:
    """Result of executing a research step."""
    step_id: str
    success: bool
    findings: str
    learning: str = ""
    next_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    code_executions: List[Dict[str, Any]] = field(default_factory=list)
    command_executions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the step result to a dictionary."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "findings": self.findings,
            "learning": self.learning,
            "next_steps": self.next_steps,
            "artifacts": self.artifacts,
            "code_executions": self.code_executions,
            "command_executions": self.command_executions,
            "error": self.error
        }
    
    def to_json(self) -> str:
        """Convert the step result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)


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
        allow_code_execution: bool = False,
        allow_command_execution: bool = False,
        allow_web_search: bool = False,
        allow_file_operations: bool = False,
        allow_data_analysis: bool = False,
        tools: Optional[List[str]] = None,
        share_output_with_llm: bool = True,  # Whether to share execution results with the LLM
        **kwargs
    ):
        """
        Initialize the StepExecutor.
        
        Args:
            service: OpenRouter service instance (created if not provided)
            model: Model to use (default: from config or deepseek/deepseek-v3-base:free)
            temperature: Default temperature for generation (default: 0.7)
            max_tokens: Default max tokens for generation (default: 2048)
            allow_code_execution: Whether to allow Python code execution (default: False)
            allow_command_execution: Whether to allow terminal command execution (default: False)
            allow_web_search: Whether to allow web search (default: False)
            allow_file_operations: Whether to allow file operations (default: False)
            allow_data_analysis: Whether to allow data analysis tools (default: False)
            tools: List of specific tools to enable (overrides individual flags)
            share_output_with_llm: Whether to share execution results with the LLM (default: True)
            **kwargs: Additional parameters for generation
        """
        # Use Deepseek as default model
        default_model = model or config_manager.get("EXECUTOR_MODEL", "google/gemini-2.0-flash-001")
        
        # Create service if not provided
        self._service = service or create_openrouter_service(model=default_model)
        
        # Store service model name instead of the actual service object
        # to avoid JSON serialization issues
        self.model_name = self._service.default_model
        
        # Store generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Store execution history
        self.execution_history: List[StepResult] = []
        
        # Initialize available tools
        if tools is not None:
            # If specific tools are provided, use those
            self.tools = tools
            self.allow_code_execution = "python" in tools or "code" in tools
            self.allow_command_execution = "terminal" in tools or "command" in tools or "shell" in tools
            self.allow_web_search = "web" in tools or "search" in tools
            self.allow_file_operations = "file" in tools
            self.allow_data_analysis = "data" in tools or "analysis" in tools
        else:
            # Otherwise use the individual flags
            self.allow_code_execution = allow_code_execution
            self.allow_command_execution = allow_command_execution
            self.allow_web_search = allow_web_search
            self.allow_file_operations = allow_file_operations
            self.allow_data_analysis = allow_data_analysis
            
            # Create list of enabled tools
            self.tools = []
            if self.allow_code_execution:
                self.tools.append("python")
            if self.allow_command_execution:
                self.tools.append("terminal")
            if self.allow_web_search:
                self.tools.append("web_search")
            if self.allow_file_operations:
                self.tools.append("file_operations")
            if self.allow_data_analysis:
                self.tools.append("data_analysis")
        
        self.share_output_with_llm = share_output_with_llm
        
        # Get logger
        self.logger = get_logger()
    
    def execute_python_code(self, code: str) -> CodeExecutionResult:
        """
        Execute Python code in a sandbox environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            CodeExecutionResult containing execution results
        """
        if not self.allow_code_execution:
            error_msg = "Code execution is disabled. Enable with allow_code_execution=True."
            self.logger.warning(
                message=error_msg,
                event_type=LogEventType.CODE_EXECUTION
            )
            return CodeExecutionResult(
                success=False,
                output="",
                error=error_msg
            )
        
        self.logger.info(
            message="Executing Python code",
            event_type=LogEventType.CODE_EXECUTION,
            context={
                "code_snippet": code[:100] + "..." if len(code) > 100 else code
            }
        )
        
        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp_path = temp.name
            try:
                # Write the code to the temporary file
                with open(temp_path, 'w') as f:
                    f.write(code)
                
                # Execute the code in a separate process and capture output
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30  # Timeout after 30 seconds
                )
                
                if result.returncode == 0:
                    self.logger.info(
                        message="Python code execution succeeded",
                        event_type=LogEventType.CODE_EXECUTION,
                        context={
                            "output_snippet": result.stdout[:100] + "..." if len(result.stdout) > 100 else result.stdout
                        }
                    )
                    return CodeExecutionResult(
                        success=True,
                        output=result.stdout,
                        error=None
                    )
                else:
                    self.logger.error(
                        message="Python code execution failed",
                        event_type=LogEventType.CODE_EXECUTION,
                        context={
                            "error": result.stderr,
                            "stdout": result.stdout
                        }
                    )
                    return CodeExecutionResult(
                        success=False,
                        output=result.stdout,
                        error=result.stderr
                    )
                    
            except subprocess.TimeoutExpired:
                error_msg = "Code execution timed out after 30 seconds"
                self.logger.error(
                    message=error_msg,
                    event_type=LogEventType.CODE_EXECUTION
                )
                return CodeExecutionResult(
                    success=False,
                    output="",
                    error=error_msg
                )
            except Exception as e:
                error_msg = f"Error executing code: {str(e)}"
                self.logger.error(
                    message=error_msg,
                    event_type=LogEventType.CODE_EXECUTION
                )
                return CodeExecutionResult(
                    success=False,
                    output="",
                    error=error_msg
                )
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    def execute_command(self, command: str) -> CommandExecutionResult:
        """
        Execute a terminal command.
        
        Args:
            command: Terminal command to execute
            
        Returns:
            CommandExecutionResult containing execution results
        """
        if not self.allow_command_execution:
            error_msg = "Command execution is disabled. Enable with allow_command_execution=True."
            self.logger.warning(
                message=error_msg,
                event_type=LogEventType.COMMAND_EXECUTION
            )
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr=error_msg,
                exit_code=1
            )
        
        # List of dangerous commands to block
        dangerous_commands = ['rm -rf', 'mkfs', 'dd', ':(){', 'chmod -R 777', '> /dev/sda']
        
        # Check if the command contains any dangerous patterns
        if any(dc in command for dc in dangerous_commands):
            error_msg = "Command execution blocked for security reasons."
            self.logger.warning(
                message=error_msg,
                event_type=LogEventType.COMMAND_EXECUTION,
                context={
                    "command": command,
                    "reason": "Security block - dangerous command pattern"
                }
            )
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr=error_msg,
                exit_code=1
            )
        
        self.logger.info(
            message=f"Executing command: {command}",
            event_type=LogEventType.COMMAND_EXECUTION
        )
        
        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # Timeout after 30 seconds
            )
            
            if result.returncode == 0:
                self.logger.info(
                    message="Command execution succeeded",
                    event_type=LogEventType.COMMAND_EXECUTION,
                    context={
                        "command": command,
                        "stdout_snippet": result.stdout[:100] + "..." if len(result.stdout) > 100 else result.stdout
                    }
                )
            else:
                self.logger.warning(
                    message=f"Command execution failed with exit code {result.returncode}",
                    event_type=LogEventType.COMMAND_EXECUTION,
                    context={
                        "command": command,
                        "stderr": result.stderr
                    }
                )
            
            return CommandExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            error_msg = "Command execution timed out after 30 seconds"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.COMMAND_EXECUTION,
                context={"command": command}
            )
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr=error_msg,
                exit_code=1
            )
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.COMMAND_EXECUTION,
                context={"command": command}
            )
            return CommandExecutionResult(
                success=False,
                stdout="",
                stderr=error_msg,
                exit_code=1
            )
    
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

        system_prompt = f"""You are an expert research assistant executing a specific step in a research plan. Your goal is to execute this step thoroughly and document your findings and learnings.

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

Execute this research step thoroughly. Think step-by-step about how to achieve the goal using the available tools.
First, analyze what needs to be done to complete this step successfully.
Then, use the available tools to gather information, perform calculations, or conduct other necessary operations.
Finally, summarize your findings and key learnings from this step.

You MUST respond in a structured JSON format with these fields:
1. "findings": Detailed results and discoveries from this step
2. "learning": Key insights or knowledge gained
3. "success": Whether the step was successful (boolean)
4. "next_steps": Array of suggestions for what to do next

When you need to use a tool, FIRST return the JSON with your current thinking, THEN indicate which tool you want to use.
You'll receive tool results that you can use to update your findings.
"""

        return system_prompt

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tool definitions for OpenRouter tool calling.
        
        Returns:
            List of tool definitions in the OpenRouter format
        """
        tools = []
        
        # Python code execution tool
        if self.allow_code_execution:
            tools.append({
                "type": "function",
                "function": {
                    "name": "python_execution",
                    "description": "Execute Python code and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of what the code does"
                            }
                        },
                        "required": ["code"]
                    }
                }
            })
        
        # Terminal command execution tool
        if self.allow_command_execution:
            tools.append({
                "type": "function",
                "function": {
                    "name": "terminal_command",
                    "description": "Execute a terminal command and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string", 
                                "description": "Terminal command to execute"
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of what the command does"
                            }
                        },
                        "required": ["command"]
                    }
                }
            })
        
        # Web search tool
        if self.allow_web_search:
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # File operations tool
        if self.allow_file_operations:
            tools.append({
                "type": "function",
                "function": {
                    "name": "file_operation",
                    "description": "Read or write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["read", "write"],
                                "description": "Operation to perform"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write (only for write operation)"
                            }
                        },
                        "required": ["operation", "path"]
                    }
                }
            })
        
        # Data analysis tool
        if self.allow_data_analysis:
            tools.append({
                "type": "function",
                "function": {
                    "name": "data_analysis",
                    "description": "Analyze data using pandas",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code using pandas for data analysis"
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of the analysis"
                            }
                        },
                        "required": ["code"]
                    }
                }
            })
        
        return tools
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on the name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        print(f"\n--- Executing tool: {tool_name} ---")
        print(f"Arguments: {json.dumps(tool_args, indent=2)}")
        
        if tool_name == "python_execution" and self.allow_code_execution:
            code = tool_args.get("code", "")
            description = tool_args.get("description", "")
            
            # Execute Python code
            result = self.execute_python_code(code)
            
            # Record execution
            execution_record = {
                "code": code,
                "output": result.output,
                "success": result.success,
                "error": result.error,
                "description": description
            }
            
            # Print output
            print(f"\n--- Python output ---")
            print(result.output if result.success else f"ERROR: {result.error}")
            
            return {
                "success": result.success,
                "output": result.output if result.success else "",
                "error": result.error if not result.success else None
            }
            
        elif tool_name == "terminal_command" and self.allow_command_execution:
            command = tool_args.get("command", "")
            description = tool_args.get("description", "")
            
            # Execute command
            result = self.execute_command(command)
            
            # Print output
            print(f"\n--- Command output (exit code: {result.exit_code}) ---")
            print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            
            return {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code
            }
            
        elif tool_name == "web_search" and self.allow_web_search:
            query = tool_args.get("query", "")
            
            # Simulate web search for now
            print(f"\n--- Web search query: {query} ---")
            search_result = f"[Simulated web search results for: {query}]"
            
            # Print output
            print(f"\n--- Search results ---")
            print(search_result)
            
            return {
                "success": True,
                "results": search_result
            }
            
        elif tool_name == "file_operation" and self.allow_file_operations:
            operation = tool_args.get("operation", "")
            path = tool_args.get("path", "")
            content = tool_args.get("content", "")
            
            # Simulate file operation for now
            print(f"\n--- File operation: {operation} on {path} ---")
            
            if operation == "read":
                try:
                    with open(path, 'r') as f:
                        file_content = f.read()
                    return {
                        "success": True,
                        "content": file_content
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            elif operation == "write":
                try:
                    with open(path, 'w') as f:
                        f.write(content)
                    return {
                        "success": True
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            return {
                "success": False,
                "error": f"Invalid operation: {operation}"
            }
            
        elif tool_name == "data_analysis" and self.allow_data_analysis:
            code = tool_args.get("code", "")
            description = tool_args.get("description", "")
            
            # Execute Python code (same as python_execution but focused on data)
            result = self.execute_python_code(code)
            
            # Print output
            print(f"\n--- Data analysis output ---")
            print(result.output if result.success else f"ERROR: {result.error}")
            
            return {
                "success": result.success,
                "output": result.output if result.success else "",
                "error": result.error if not result.success else None
            }
        
        return {
            "success": False,
            "error": f"Tool not available or not allowed: {tool_name}"
        }

    async def execute_step_async(
        self,
        step: ResearchStep,
        research_plan: ResearchPlan,
        previous_results: Optional[List[StepResult]] = None,
        max_interactions: int = 5,
        additional_context: Optional[str] = None,
        allow_code_execution: bool = False,
        allow_command_execution: bool = False,
        allow_web_search: bool = False,
        allow_file_operations: bool = False,
        allow_data_analysis: bool = False,
        agent_id_for_logs: Optional[str] = None,
        task_id_for_logs: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0, 
        tools: Optional[List[Union[str, Tool]]] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a single research step.
        
        Args:
            step: The step to execute
            research_plan: The overall research plan
            previous_results: Results from previously executed steps
            max_interactions: Maximum number of interactions for the step
            additional_context: Any additional context to provide
            allow_code_execution: Whether to allow Python code execution
            allow_command_execution: Whether to allow terminal command execution
            allow_web_search: Whether to allow web search
            allow_file_operations: Whether to allow file operations
            allow_data_analysis: Whether to allow data analysis
            agent_id_for_logs: Agent ID for logging
            task_id_for_logs: Task ID for logging
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Initial delay between retries in seconds
            tools: List of specific tools to enable
            **kwargs: Additional parameters for execution
            
        Returns:
            StepResult containing the findings and learnings from the step
        """
        try:
            # Set IDs for logging
            agent_id = agent_id_for_logs
            task_id = task_id_for_logs
            
            # Log step execution start
            self.logger.info(
                message=f"Executing step: {step.id} - {step.title}",
                event_type=LogEventType.AGENT_TASK_STARTED,
                agent_id=agent_id,
                task_id=task_id
            )
            
            # Print step information
            print(f"\n==== EXECUTING STEP: {step.id} - {step.title} ====")
            print(f"Goal: {step.goal}")
            
            # Get available tools
            tool_instances = []
            
            # Figure out which tools to use
            if tools is not None:
                # If specific tools are provided, use those
                for tool in tools:
                    if isinstance(tool, str):
                        if tool.lower() in ["python", "code"]:
                            allow_code_execution = True
                        elif tool.lower() in ["terminal", "command", "shell"]:
                            allow_command_execution = True
                        elif tool.lower() in ["web", "search"]:
                            allow_web_search = True
                        elif tool.lower() in ["file"]:
                            allow_file_operations = True
                        elif tool.lower() in ["data", "analysis"]:
                            allow_data_analysis = True
                    elif isinstance(tool, Tool):
                        tool_instances.append(tool)
            
            # Create prompt for step execution
            prompt, tool_definitions = self._create_step_execution_prompt(
                step=step,
                research_plan=research_plan,
                previous_results=previous_results,
                additional_context=additional_context
            )
            
            # Configure the tools allowed
            active_tools = []
            if allow_code_execution:
                active_tools.append("python_code")
                print("🐍 Enabled Python code execution")
            if allow_command_execution:
                active_tools.append("terminal_command")
                print("🖥️ Enabled terminal command execution")
            if allow_web_search:
                active_tools.append("web_search")
                print("🔍 Enabled web search")
            if allow_file_operations:
                active_tools.append("file_operations")
                print("📁 Enabled file operations")
            if allow_data_analysis:
                active_tools.append("data_analysis")
                print("📊 Enabled data analysis")
            
            print(f"Available tools: {active_tools}")
            
            # Initialize variables to track execution
            interactions = 0
            findings = ""
            learning = ""
            next_steps = []
            success = False
            all_tool_calls = []
            
            # Get model parameters with defaults from adapter
            params = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            
            # Override with any function-specific kwargs
            for key, value in kwargs.items():
                if key not in ["temperature", "max_tokens"]:
                    params[key] = value
            
            # Add the tool definitions if we have any active tools
            if active_tools:
                params["tools"] = self._get_tool_definitions()
            
            # Main interaction loop
            retry_count = 0
            while interactions < max_interactions:
                interactions += 1
                
                # Format the user message for this interaction
                if interactions == 1:
                    # First interaction: provide the task
                    user_message = "Execute this research step and provide findings."
                else:
                    # Subsequent interactions: continue from previous context
                    user_message = "Continue working on this step based on the results so far."
                
                try:
                    # Create messages for this interaction
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    # Make API call with retry logic for connection errors
                    retry_attempt = 0
                    while True:
                        try:
                            response = await self._service.generate_chat_response(
                                messages=messages,
                                **params
                            )
                            break  # Exit retry loop if successful
                            
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            # Handle connection errors with retries
                            error_type = type(e).__name__
                            retry_attempt += 1
                            
                            # Log the error
                            self.logger.error(
                                f"Connection error ({error_type}) in LLM call for step {step.id} (attempt {retry_attempt}/{max_retries}): {str(e)}"
                            )
                            
                            if retry_attempt < max_retries:
                                # Wait with exponential backoff
                                wait_time = retry_delay * (2 ** retry_attempt)
                                self.logger.info(f"Retrying LLM call in {wait_time:.1f} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                # Max retries exceeded
                                self.logger.error(f"Max retries ({max_retries}) exceeded for LLM call")
                                raise
                    
                    # Print debug info for raw response
                    print(f"\n--- Interaction {interactions}/{max_interactions} ---\n")
                    print("--- RAW LLM RESPONSE ---")
                    print(json.dumps(response, indent=2))
                    print("------------------------")
                    
                    # Extract assistant message
                    if "choices" in response and len(response["choices"]) > 0:
                        assistant_message = response["choices"][0]["message"]
                        
                        # Print assistant message for debugging
                        print("\n--- ASSISTANT MESSAGE ---")
                        print(json.dumps(assistant_message, indent=2))
                        print("--------------------------")
                        
                        content = assistant_message.get("content", "")
                        tool_calls = assistant_message.get("tool_calls", [])
                        
                        print("\n--- FINAL CONTENT ---")
                        print(content)
                        print("---------------------")
                        
                        # Execute any tool calls
                        if tool_calls:
                            for tool_call in tool_calls:
                                # Extract tool call info
                                function = tool_call.get("function", {})
                                tool_name = function.get("name", "")
                                arguments_str = function.get("arguments", "{}")
                                
                                try:
                                    # Parse arguments
                                    arguments = json.loads(arguments_str)
                                    
                                    # Log tool call
                                    print(f"\n--- TOOL CALL: {tool_name} ---")
                                    print(f"Arguments: {json.dumps(arguments, indent=2)}")
                                    
                                    # Execute the tool
                                    tool_result = await self._execute_tool(
                                        tool_name=tool_name,
                                        arguments=arguments,
                                        allow_code_execution=allow_code_execution,
                                        allow_command_execution=allow_command_execution,
                                        allow_web_search=allow_web_search,
                                        allow_file_operations=allow_file_operations,
                                        allow_data_analysis=allow_data_analysis
                                    )
                                    
                                    # Log result
                                    print(f"Result: {tool_result}\n")
                                    
                                    # Add tool call to history
                                    all_tool_calls.append({
                                        "tool": tool_name,
                                        "args": arguments,
                                        "result": tool_result
                                    })
                                    
                                    # Add tool result to messages for context
                                    messages.append({
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [tool_call]
                                    })
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.get("id", ""),
                                        "content": str(tool_result)
                                    })
                                    
                                except Exception as e:
                                    # Log tool execution error
                                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                                    self.logger.error(error_msg)
                                    print(f"Error: {error_msg}")
                                    
                                    # Add error message as tool result
                                    messages.append({
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [tool_call]
                                    })
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call.get("id", ""),
                                        "content": f"Error: {str(e)}"
                                    })
                            
                            # Get a new response after tool execution
                            # Use retry logic for connection errors
                            retry_attempt = 0
                            while True:
                                try:
                                    response = await self._service.generate_chat_response(
                                        messages=messages,
                                        **params
                                    )
                                    break  # Exit retry loop if successful
                                    
                                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                                    # Handle connection errors with retries
                                    error_type = type(e).__name__
                                    retry_attempt += 1
                                    
                                    # Log the error
                                    self.logger.error(
                                        f"Connection error ({error_type}) in follow-up LLM call (attempt {retry_attempt}/{max_retries}): {str(e)}"
                                    )
                                    
                                    if retry_attempt < max_retries:
                                        # Wait with exponential backoff
                                        wait_time = retry_delay * (2 ** retry_attempt)
                                        self.logger.info(f"Retrying follow-up LLM call in {wait_time:.1f} seconds...")
                                        await asyncio.sleep(wait_time)
                                    else:
                                        # Max retries exceeded
                                        self.logger.error(f"Max retries ({max_retries}) exceeded for follow-up LLM call")
                                        raise
                            
                            # Extract content from the new response
                            if "choices" in response and len(response["choices"]) > 0:
                                content = response["choices"][0]["message"].get("content", "")
                                print("\n--- FINAL CONTENT AFTER TOOL CALLS ---")
                                print(content)
                                print("-------------------------------------")
                        
                        # Try to parse the content as JSON to extract results
                        try:
                            # First check if the entire content is a JSON object
                            try:
                                parsed_json = json.loads(content)
                                print("\n--- JSON PARSED RESULT ---")
                                print(json.dumps(parsed_json, indent=2))
                                
                                # Extract results from parsed JSON
                                if isinstance(parsed_json, dict):
                                    if "findings" in parsed_json:
                                        findings = parsed_json.get("findings", "")
                                    if "learning" in parsed_json:
                                        learning = parsed_json.get("learning", "")
                                    if "next_steps" in parsed_json:
                                        next_steps = parsed_json.get("next_steps", [])
                                    if "success" in parsed_json:
                                        success = parsed_json.get("success", False)
                            
                            except json.JSONDecodeError:
                                # If the content is not a JSON object, try to extract JSON from a code block
                                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                                if json_match:
                                    try:
                                        parsed_json = json.loads(json_match.group(1))
                                        print("\n--- JSON BLOCK EXTRACTED FROM ```json CODE BLOCK ---")
                                        print(json_match.group(1))
                                        print("\n--- PARSED JSON RESULT ---")
                                        print(json.dumps(parsed_json, indent=2))
                                        
                                        # Extract results from parsed JSON
                                        if isinstance(parsed_json, dict):
                                            if "findings" in parsed_json:
                                                findings = parsed_json.get("findings", "")
                                            if "learning" in parsed_json:
                                                learning = parsed_json.get("learning", "")
                                            if "next_steps" in parsed_json:
                                                next_steps = parsed_json.get("next_steps", [])
                                            if "success" in parsed_json:
                                                success = parsed_json.get("success", False)
                                    except json.JSONDecodeError as e:
                                        print(f"Error parsing JSON from code block: {e}")
                                else:
                                    # If we couldn't find a JSON code block, use text parsing as fallback
                                    print("\n--- TEXT PARSED RESULT ---")
                                    parsed_text = {
                                        "success": "success" in content.lower() and "fail" not in content.lower(),
                                        "findings": content,
                                        "learning": "Learned information about this research step.",
                                        "next_steps": []
                                    }
                                    print(json.dumps(parsed_text, indent=2))
                                    
                                    # Extract results from text parsing
                                    findings = parsed_text["findings"]
                                    learning = parsed_text["learning"]
                                    success = parsed_text["success"]
                        
                        except Exception as e:
                            self.logger.error(f"Error parsing result: {str(e)}")
                            print(f"Error parsing result: {str(e)}")
                            
                            # Use the raw content as findings
                            findings = content
                            learning = "Error parsing the structured output."
                            success = False
                    
                    # Check if we have enough information to complete the step
                    if findings and (success or interactions >= max_interactions):
                        break
                
                except Exception as e:
                    self.logger.error(f"Error in step execution: {str(e)}")
                    print(f"Error in step execution: {str(e)}")
                    
                    # If we've reached max retries, create an error result
                    if retry_count >= max_retries:
                        return StepResult(
                            step_id=step.id,
                            success=False,
                            findings=f"Error executing step: {str(e)}",
                            learning="Error handling is important for robust research workflows.",
                            error=str(e),
                            tool_calls=all_tool_calls
                        )
                    
                    # Otherwise, increment retry count and continue
                    retry_count += 1
                    
                    # Wait with exponential backoff
                    wait_time = retry_delay * (2 ** retry_count)
                    self.logger.info(f"Retrying step execution in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
            
            # Create step result
            result = StepResult(
                step_id=step.id,
                success=success,
                findings=findings,
                learning=learning,
                next_steps=next_steps,
                tool_calls=all_tool_calls
            )
            
            # Print execution results
            print("\n==== STEP EXECUTION RESULTS ====")
            print(f"Success: {result.success}")
            print(f"Findings: {result.findings[:120]}...")
            print(f"Learning: {result.learning}")
            print(f"Next Steps: {result.next_steps}")
            print("================================\n")
            
            return result
        
        finally:
            # Ensure any sessions created during step execution are properly closed
            if hasattr(self, "_service") and hasattr(self._service, "_close_session"):
                try:
                    await self._service._close_session()
                except Exception as e:
                    self.logger.warning(f"Error closing service session: {str(e)}")
                    
            # Close any sessions from tools
            if hasattr(self, "web_search_tool") and hasattr(self.web_search_tool, "_close_session"):
                try:
                    await self.web_search_tool._close_session()
                except Exception as e:
                    self.logger.warning(f"Error closing web search tool session: {str(e)}")


# Convenience function to create a step executor
def create_step_executor(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    allow_code_execution: bool = False,
    allow_command_execution: bool = False
) -> StepExecutor:
    """
    Create a StepExecutor instance with configuration.
    
    Args:
        api_key: OpenRouter API key (optional)
        model: Model to use (default: from config or claude)
        temperature: Temperature for generation (default: 0.7)
        max_tokens: Max tokens for generation (default: 2048)
        allow_code_execution: Whether to allow Python code execution (default: False)
        allow_command_execution: Whether to allow terminal command execution (default: False)
        
    Returns:
        Configured StepExecutor instance
    """
    # Create the service
    service = create_openrouter_service(api_key=api_key, model=model)
    
    # Create and return the executor
    return StepExecutor(
        service=service,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_code_execution=allow_code_execution,
        allow_command_execution=allow_command_execution
    ) 