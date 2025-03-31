"""
StepExecutor module for executing individual research steps from a research plan.
"""
import os
import sys
import json
import subprocess
import tempfile
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
            model: Model to use (default: from config or google/gemini-2.0-flash-001)
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
        # Use Claude as default model for better reasoning
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
        additional_context: Optional[str] = None,
        step_id_for_logs: Optional[str] = None,
        task_id_for_logs: Optional[str] = None,
        agent_id_for_logs: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StepResult:
        """
        Execute a step of the research plan asynchronously.
        
        Args:
            step: Research step to execute
            research_plan: The full research plan
            previous_results: Results from previous steps
            additional_context: Additional context for execution
            step_id_for_logs: Optional step ID for logging
            task_id_for_logs: Optional task ID for logging
            agent_id_for_logs: Optional agent ID for logging
            temperature: Temperature for generation
            max_tokens: Max tokens for generation
            **kwargs: Additional parameters for generation
            
        Returns:
            StepResult object with execution results
        """
        # Use provided temperature/max_tokens or fall back to defaults
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Create prompt for the step execution
        system_prompt = self._create_step_execution_prompt(
            step=step,
            research_plan=research_plan,
            previous_results=previous_results,
            additional_context=additional_context
        )
        
        # Create task prompt
        user_prompt = f"Execute research step {step.id}: {step.title}"
        
        # Get tool definitions
        tools = self._get_tool_definitions()
        
        # Initialize execution records
        code_executions = []
        command_executions = []
        web_search_results = []
        file_operations = []
        data_analysis_results = []
        
        # Initialize step result
        step_result = StepResult(
            step_id=step.id,
            success=False,
            findings="",
            learning="",
            code_executions=code_executions,
            command_executions=command_executions,
            artifacts={
                "web_search_results": web_search_results,
                "file_operations": file_operations,
                "data_analysis_results": data_analysis_results
            }
        )
        
        try:
            # Log step execution
            if agent_id_for_logs and task_id_for_logs:
                self.logger.info(
                    message=f"Executing step: {step.id} - {step.title}",
                    agent_id=agent_id_for_logs,
                    task_id=task_id_for_logs,
                    context={
                        "step_goal": step.goal,
                        "prompt_length": len(system_prompt),
                        "available_tools": self.tools
                    }
                )
            
            # Print debug information
            print(f"\n==== EXECUTING STEP: {step.id} - {step.title} ====")
            print(f"Goal: {step.goal}")
            print(f"Available tools: {self.tools}")
            
            # Create messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Create execution parameters without any JSON-problematic objects
            execution_params = {
                "model": self.model_name,  # Use model name string, not the service object
                "temperature": temp,
                "max_tokens": tokens
            }
            
            # Only add tools parameter if we have tools defined
            if tools:
                execution_params["tools"] = tools
                
            # Filter kwargs to remove any non-serializable objects
            filtered_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    filtered_kwargs[key] = value
            
            # Add filtered kwargs to execution params
            execution_params.update(filtered_kwargs)
            
            # Continue conversation until step is complete or max iterations reached
            max_iterations = 5
            current_iteration = 0
            final_result = None
            
            while current_iteration < max_iterations:
                current_iteration += 1
                print(f"\n--- Interaction {current_iteration}/{max_iterations} ---")
                
                # Call LLM to execute the step with safe parameters
                response = await self._service.generate_chat_response(
                    messages=messages,
                    **execution_params
                )
                
                # Print the raw response for debugging
                print("\n--- RAW LLM RESPONSE ---")
                print(json.dumps(response, indent=2, default=str))
                print("------------------------")
                
                # Extract assistant's message
                assistant_message = None
                if "choices" in response and len(response["choices"]) > 0:
                    assistant_message = response["choices"][0]["message"]
                
                if not assistant_message:
                    raise ValueError("Empty response from LLM")
                
                # Print assistant's message for debugging
                print("\n--- ASSISTANT MESSAGE ---")
                print(json.dumps(assistant_message, indent=2, default=str))
                print("--------------------------")
                
                # Add assistant's message to conversation
                messages.append(assistant_message)
                
                # Check if the assistant wants to use a tool
                tool_calls = assistant_message.get("tool_calls", [])
                
                if not tool_calls:
                    # No tool calls, this is the final response
                    final_result = assistant_message
                    break
                
                # Process tool calls
                for tool_call in tool_calls:
                    # Extract tool call information
                    if tool_call.get("type") == "function":
                        function_call = tool_call.get("function", {})
                        tool_name = function_call.get("name")
                        tool_args_str = function_call.get("arguments", "{}")
                        tool_args = json.loads(tool_args_str)
                        
                        # Print tool call for debugging
                        print(f"\n--- TOOL CALL: {tool_name} ---")
                        print(f"Arguments: {json.dumps(tool_args, indent=2)}")
                        
                        # Execute the tool
                        tool_result = await self._execute_tool(tool_name, tool_args)
                        
                        # Print tool result for debugging
                        print(f"\n--- TOOL RESULT ---")
                        print(json.dumps(tool_result, indent=2))
                        print("-------------------")
                        
                        # Record tool execution based on tool type
                        if tool_name == "python_execution":
                            code_executions.append({
                                "code": tool_args.get("code", ""),
                                "output": tool_result.get("output", ""),
                                "success": tool_result.get("success", False),
                                "error": tool_result.get("error"),
                                "description": tool_args.get("description", "")
                            })
                        elif tool_name == "terminal_command":
                            command_executions.append({
                                "command": tool_args.get("command", ""),
                                "stdout": tool_result.get("stdout", ""),
                                "stderr": tool_result.get("stderr", ""),
                                "exit_code": tool_result.get("exit_code", 1),
                                "success": tool_result.get("success", False),
                                "description": tool_args.get("description", "")
                            })
                        elif tool_name == "web_search":
                            web_search_results.append({
                                "query": tool_args.get("query", ""),
                                "results": tool_result.get("results", "")
                            })
                        elif tool_name == "file_operation":
                            file_operations.append({
                                "operation": tool_args.get("operation", ""),
                                "path": tool_args.get("path", ""),
                                "success": tool_result.get("success", False),
                                "error": tool_result.get("error")
                            })
                        elif tool_name == "data_analysis":
                            data_analysis_results.append({
                                "code": tool_args.get("code", ""),
                                "output": tool_result.get("output", ""),
                                "success": tool_result.get("success", False),
                                "error": tool_result.get("error"),
                                "description": tool_args.get("description", "")
                            })
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": tool_name,
                            "content": json.dumps(tool_result)
                        })
            
            # If we didn't get a final result, use the last message
            if not final_result and messages:
                for message in reversed(messages):
                    if message.get("role") == "assistant":
                        final_result = message
                        break
            
            if not final_result:
                raise ValueError("No final result from LLM")
            
            # Extract the content from the final result
            content = final_result.get("content", "")
            
            # Print final content for debugging
            print("\n--- FINAL CONTENT ---")
            print(content)
            print("---------------------")
            
            # Try to parse JSON from content
            try:
                # Check if it's already a JSON object
                if isinstance(content, dict):
                    result_data = content
                else:
                    # Try to extract JSON from the text
                    json_match = None
                    if "```json" in content:
                        parts = content.split("```json")
                        if len(parts) > 1:
                            json_block = parts[1].split("```")[0].strip()
                            json_match = json.loads(json_block)
                            print("\n--- JSON BLOCK EXTRACTED FROM ```json CODE BLOCK ---")
                            print(json_block)
                    elif content.strip().startswith("{") and content.strip().endswith("}"):
                        json_match = json.loads(content)
                        print("\n--- JSON PARSED FROM RAW CONTENT ---")
                        print(content)
                    
                    if json_match:
                        result_data = json_match
                        print("\n--- PARSED JSON RESULT ---")
                        print(json.dumps(result_data, indent=2))
                    else:
                        # Fall back to parsing with our text extraction method
                        result_data = self._parse_step_result(content, step.id)
                        print("\n--- TEXT PARSED RESULT ---")
                        print(json.dumps(result_data, indent=2))
            except Exception as e:
                print(f"\n--- JSON PARSING ERROR ---")
                print(f"Error: {str(e)}")
                result_data = self._parse_step_result(content, step.id)
                print("\n--- FALLBACK TEXT PARSED RESULT ---")
                print(json.dumps(result_data, indent=2))
            
            # Update the step result
            step_result.success = result_data.get("success", False)
            step_result.findings = result_data.get("findings", "")
            step_result.learning = result_data.get("learning", "")
            step_result.next_steps = result_data.get("next_steps", [])
            
            # Print the results
            print(f"\n==== STEP EXECUTION RESULTS ====")
            print(f"Success: {step_result.success}")
            print(f"Findings: {step_result.findings[:200]}..." if len(step_result.findings) > 200 else f"Findings: {step_result.findings}")
            print(f"Learning: {step_result.learning}")
            print(f"Next Steps: {step_result.next_steps}")
            print("================================\n")
            
            # Log completion
            if agent_id_for_logs and task_id_for_logs:
                self.logger.info(
                    message=f"Step {step.id} execution completed",
                    agent_id=agent_id_for_logs,
                    task_id=task_id_for_logs,
                    context={
                        "success": step_result.success,
                        "findings_length": len(step_result.findings),
                        "learning_length": len(step_result.learning),
                        "tools_used": {
                            "code": len(code_executions),
                            "commands": len(command_executions),
                            "web_searches": len(web_search_results),
                            "file_operations": len(file_operations),
                            "data_analyses": len(data_analysis_results)
                        }
                    }
                )
            
            # Add to execution history
            self.execution_history.append(step_result)
            
            # Return the result
            return step_result
                
        except Exception as e:
            error_msg = f"Error during step execution: {str(e)}"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.ERROR
            )
            
            # Print error
            print(f"\n==== ERROR IN STEP EXECUTION ====")
            print(error_msg)
            print("================================\n")
            
            # Update step result with error
            step_result.success = False
            step_result.error = error_msg
            
            # Add to execution history
            self.execution_history.append(step_result)
            
            return step_result
    
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
    
    async def execute_plan_async(
        self,
        research_plan: ResearchPlan,
        additional_context: Optional[str] = None,
        agent_id_for_logs: Optional[str] = None,
        task_id_for_logs: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[StepResult]:
        """
        Execute all steps in a research plan, respecting dependencies.
        
        Args:
            research_plan: The research plan to execute
            additional_context: Any additional context (optional)
            agent_id_for_logs: Agent ID for logging (optional)
            task_id_for_logs: Task ID for logging (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            List of StepResult objects for all executed steps
        """
        # Set IDs for logging
        agent_id = agent_id_for_logs
        task_id = task_id_for_logs
        
        self.logger.info(
            message=f"Executing research plan: {research_plan.title}",
            event_type=LogEventType.PLAN_GENERATION_COMPLETED,
            agent_id=agent_id,
            task_id=task_id,
            context={
                "steps_count": len(research_plan.steps),
                "objective": research_plan.objective
            }
        )
        
        # Reset execution history
        self.execution_history = []
        
        # Track completed steps by ID
        completed_steps: Dict[str, StepResult] = {}
        
        # Process steps in order, respecting dependencies
        for step in research_plan.steps:
            # Check if all dependencies are completed
            if not all(dep_id in completed_steps for dep_id in step.dependencies):
                self.logger.warning(
                    message=f"Skipping step {step.id} due to missing dependencies",
                    event_type=LogEventType.STEP_EXECUTION_STARTED,
                    agent_id=agent_id,
                    task_id=task_id,
                    context={
                        "step_id": step.id,
                        "dependencies": step.dependencies,
                        "completed_steps": list(completed_steps.keys())
                    }
                )
                
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
            result = await self.execute_step_async(
                step=step,
                research_plan=research_plan,
                previous_results=dependency_results if dependency_results else None,
                additional_context=additional_context,
                agent_id_for_logs=agent_id,
                task_id_for_logs=task_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Store the result
            completed_steps[step.id] = result
        
        self.logger.info(
            message=f"Research plan execution completed with {len([r for r in self.execution_history if r.success])}/{len(self.execution_history)} successful steps",
            event_type=LogEventType.PLAN_GENERATION_COMPLETED,
            agent_id=agent_id,
            task_id=task_id
        )
        
        return self.execution_history

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from a string.
        
        Args:
            content: Text content containing code blocks
            
        Returns:
            List of extracted code blocks with type and code
        """
        blocks = []
        lines = content.split('\n')
        
        in_block = False
        current_block = {"type": "", "code": "", "description": ""}
        block_description = ""
        
        for line in lines:
            # Check for block start
            if not in_block and "```" in line:
                in_block = True
                block_type = line.replace("```", "").strip().lower()
                if block_type:
                    current_block["type"] = block_type
                else:
                    current_block["type"] = "code"  # Default block type
                
                # Look for description in previous lines
                if block_description:
                    current_block["description"] = block_description
                
                continue
            
            # Check for block end
            if in_block and "```" in line:
                in_block = False
                blocks.append(current_block)
                current_block = {"type": "", "code": "", "description": ""}
                block_description = ""
                continue
            
            # Collect code if inside a block
            if in_block:
                current_block["code"] += line + "\n"
            # Collect potential description before a code block
            elif not in_block and line.strip():
                block_description = line.strip()
        
        # Clean up code blocks
        for block in blocks:
            block["code"] = block["code"].strip()
        
        return blocks
    
    def _parse_step_result(self, content: str, step_id: str) -> Dict[str, Any]:
        """
        Parse the step execution result from the LLM output.
        
        Args:
            content: LLM output content
            step_id: ID of the step being executed
            
        Returns:
            Dictionary with parsed findings, learnings, success, next steps
        """
        result = {
            "success": False,
            "findings": "",
            "learning": "",
            "next_steps": []
        }
        
        # Try to find structured data in content
        try:
            # Look for JSON block first
            json_match = None
            if "```json" in content:
                parts = content.split("```json")
                if len(parts) > 1:
                    json_block = parts[1].split("```")[0].strip()
                    json_match = json.loads(json_block)
            elif "```" in content:
                # Try to find any code block that might contain JSON
                blocks = self._extract_code_blocks(content)
                for block in blocks:
                    try:
                        json_match = json.loads(block["code"])
                        if isinstance(json_match, dict):
                            break
                    except:
                        continue
            
            # If we found JSON, extract fields
            if json_match and isinstance(json_match, dict):
                result["findings"] = json_match.get("findings", "")
                result["learning"] = json_match.get("learning", "")
                result["success"] = json_match.get("success", False)
                result["next_steps"] = json_match.get("next_steps", [])
                return result
        except:
            # If JSON parsing failed, continue with text parsing
            pass
        
        # Try to extract structured information from text
        sections = {
            "findings": ["findings:", "results:", "observations:"],
            "learning": ["learning:", "learnings:", "lessons:", "what we learned:"],
            "success": ["success:", "successful:"],
            "next_steps": ["next steps:", "next:", "future work:", "what's next:"]
        }
        
        # Split content into lines for analysis
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line starts a new section
            found_section = False
            for section, markers in sections.items():
                for marker in markers:
                    if line_lower.startswith(marker):
                        current_section = section
                        # Extract content after the marker
                        section_content = line[len(marker):].strip()
                        
                        # Special handling for success section
                        if section == "success":
                            result[section] = "yes" in section_content.lower() or "true" in section_content.lower()
                        # Special handling for next steps as a list
                        elif section == "next_steps" and section_content:
                            if not result[section]:
                                result[section] = []
                            result[section].append(section_content)
                        # For other sections, start collecting content
                        elif section_content:
                            if result[section]:
                                result[section] += "\n" + section_content
                            else:
                                result[section] = section_content
                        
                        found_section = True
                        break
                if found_section:
                    break
            
            # If not a section header and we're in a section, append content
            if not found_section and current_section and line.strip():
                if current_section == "next_steps":
                    # Check if it's a list item
                    if line.strip().startswith("- ") or line.strip().startswith("* "):
                        result[current_section].append(line.strip()[2:])
                    elif line.strip().startswith("1. ") or line.strip().startswith("2. "):
                        result[current_section].append(line.strip()[3:])
                    else:
                        # Append to the last next step
                        if result[current_section] and len(result[current_section]) > 0:
                            result[current_section][-1] += " " + line.strip()
                else:
                    if result[current_section]:
                        result[current_section] += "\n" + line.strip()
                    else:
                        result[current_section] = line.strip()
        
        # If no structured data was found, use the whole content as findings
        if not result["findings"]:
            result["findings"] = content
        
        # If no explicit success indicator, default to True
        if not result.get("success", None):
            # Look for failure indicators
            failure_indicators = ["error", "fail", "unsuccessful", "didn't work", "did not work"]
            for indicator in failure_indicators:
                if indicator in content.lower():
                    result["success"] = False
                    break
            else:
                result["success"] = True
        
        # Default empty learning if none found
        if not result["learning"]:
            # Try to derive learning from findings
            if result["findings"]:
                result["learning"] = "Learned information about this research step."
        
        # Ensure next_steps is a list
        if not isinstance(result["next_steps"], list):
            if result["next_steps"]:
                result["next_steps"] = [result["next_steps"]]
            else:
                result["next_steps"] = []
        
        return result


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