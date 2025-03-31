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
    learning: str
    next_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    code_executions: List[Dict[str, Any]] = field(default_factory=list)
    command_executions: List[Dict[str, Any]] = field(default_factory=list)
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
        allow_code_execution: bool = False,
        allow_command_execution: bool = False,
        share_output_with_llm: bool = True,  # Whether to share execution results with the LLM
        **kwargs
    ):
        """
        Initialize the StepExecutor.
        
        Args:
            service: OpenRouter service instance (created if not provided)
            model: Model to use (default: from config or anthropic/claude-3-opus)
            temperature: Default temperature for generation (default: 0.7)
            max_tokens: Default max tokens for generation (default: 2048)
            allow_code_execution: Whether to allow Python code execution (default: False)
            allow_command_execution: Whether to allow terminal command execution (default: False)
            share_output_with_llm: Whether to share execution results with the LLM (default: True)
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
        
        # Security flags
        self.allow_code_execution = allow_code_execution
        self.allow_command_execution = allow_command_execution
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
        
        # Add execution capabilities information
        capabilities_info = "\n\n## EXECUTION CAPABILITIES\n"
        if self.allow_code_execution:
            capabilities_info += """You can execute Python code by including it in your response using the following format:
```execute_python
# Your Python code here
import pandas as pd
data = {'Name': ['John', 'Anna'], 'Age': [28, 34]}
df = pd.DataFrame(data)
print(df)
```

"""
            if self.share_output_with_llm:
                capabilities_info += "The code will be executed, and the results will be shared back with you to analyze.\n"
            else:
                capabilities_info += "The code will be executed, but results will not be shared back with you.\n"
        
        if self.allow_command_execution:
            capabilities_info += """You can execute terminal commands by including them in your response using the following format:
```execute_command
ls -la
```

"""
            if self.share_output_with_llm:
                capabilities_info += "The command will be executed, and the results will be shared back with you to analyze.\n"
            else:
                capabilities_info += "The command will be executed, but results will not be shared back with you.\n"
        
        if not self.allow_code_execution and not self.allow_command_execution:
            capabilities_info = ""  # Remove capabilities section if nothing is allowed
        
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
Expected Output: {step.expected_output}{dependency_info}{context_info}{capabilities_info}

Execute this research step thoroughly. Think step-by-step about how to achieve the goal. 
Your response must be in the following JSON format:

{{
  "findings": "Detailed findings from executing this step. Include all relevant information discovered.",
  "learning": "The key learning or insight gained from this step, summarized concisely.",
  "next_steps": ["Suggestion for next step 1", "Suggestion for next step 2"],
  "artifacts": {{
    "notes": "Any additional notes or information",
    "references": ["Reference 1", "Reference 2"]
  }}"""
        
        if self.allow_code_execution:
            system_prompt += """,
  "code_blocks": [
    {
      "code": "import pandas as pd\\ndf = pd.DataFrame({'A': [1, 2]})\\nprint(df)",
      "description": "Analysis of dataset with pandas"
    }
  ]"""
        
        if self.allow_command_execution:
            system_prompt += """,
  "command_blocks": [
    {
      "command": "ls -la",
      "description": "List files in the current directory"
    }
  ]"""
        
        system_prompt += "\n}\n\n"
        
        system_prompt += """Make sure your findings directly address the goal of this step and provide the expected output.
The 'learning' field should capture the most important insight that should be carried forward to dependent steps.

If you use code or command execution, include the code or commands in the appropriate blocks in your JSON response.
"""
        return system_prompt
    
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
        Execute a research step asynchronously.
        
        Args:
            step: The research step to execute
            research_plan: The overall research plan
            previous_results: Results from previous steps (optional)
            additional_context: Any additional context (optional)
            step_id_for_logs: Step ID for logging (optional)
            task_id_for_logs: Task ID for logging (optional)
            agent_id_for_logs: Agent ID for logging (optional)
            temperature: Temperature for generation (overrides default)
            max_tokens: Max tokens for generation (overrides default)
            **kwargs: Additional parameters for generation
            
        Returns:
            A StepResult object containing the execution results
        """
        # Set IDs for logging
        agent_id = agent_id_for_logs
        task_id = task_id_for_logs
        step_id = step_id_for_logs or step.id
        
        self.logger.info(
            message=f"Executing step: {step.title}",
            event_type=LogEventType.STEP_EXECUTION_STARTED,
            agent_id=agent_id,
            task_id=task_id,
            context={
                "step_id": step_id,
                "step_goal": step.goal,
                "dependencies": step.dependencies
            }
        )
        
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
        
        self.logger.debug(
            message="Sending step execution request to LLM",
            event_type=LogEventType.STEP_EXECUTION_STARTED,
            agent_id=agent_id,
            task_id=task_id,
            context={
                "step_id": step_id,
                "model": self.service.model,
                "temperature": params["temperature"]
            }
        )
        
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
                    
                    self.logger.info(
                        message="Successfully parsed LLM response as JSON",
                        event_type=LogEventType.STEP_EXECUTION_COMPLETED,
                        agent_id=agent_id,
                        task_id=task_id,
                        context={
                            "step_id": step_id
                        }
                    )
                    
                    # Execute any code blocks
                    code_execution_results = []
                    if self.allow_code_execution and "code_blocks" in result_data:
                        all_code_outputs = []
                        
                        for i, code_block in enumerate(result_data.get("code_blocks", [])):
                            code = code_block.get("code", "")
                            description = code_block.get("description", "")
                            
                            if code:
                                self.logger.info(
                                    message=f"Executing code block {i+1}: {description}",
                                    event_type=LogEventType.CODE_EXECUTION,
                                    agent_id=agent_id,
                                    task_id=task_id,
                                    context={
                                        "step_id": step_id,
                                        "code_snippet": code[:100] + "..." if len(code) > 100 else code
                                    }
                                )
                                
                                # Execute the code
                                execution_result = self.execute_python_code(code)
                                
                                # Store execution result
                                code_exec_result = {
                                    "code": code,
                                    "description": description,
                                    "success": execution_result.success,
                                    "output": execution_result.output,
                                    "error": execution_result.error
                                }
                                code_execution_results.append(code_exec_result)
                                
                                # Add to outputs for sharing with LLM if enabled
                                if self.share_output_with_llm:
                                    output_info = f"Code block {i+1} ({description}) result:\n"
                                    if execution_result.success:
                                        output_info += f"```\n{execution_result.output}\n```"
                                    else:
                                        output_info += f"Error: {execution_result.error}"
                                    all_code_outputs.append(output_info)
                        
                        # If sharing is enabled and we have outputs, send them to the LLM
                        if self.share_output_with_llm and all_code_outputs:
                            code_results_message = "\n\n".join(all_code_outputs)
                            messages.append({"role": "user", "content": f"Here are the results of your code execution:\n\n{code_results_message}\n\nPlease analyze these results and continue with your findings."})
                            
                            # Make another API call with the updated messages and execution results
                            updated_response = await self.service.generate_chat_response(
                                messages=messages,
                                **params
                            )
                            
                            # Update response with the new content that includes analysis of code execution results
                            if "choices" in updated_response and len(updated_response["choices"]) > 0:
                                response = updated_response
                                content = response["choices"][0]["message"]["content"]
                                
                                # Reparse the JSON from the updated response
                                if "```json" in content:
                                    json_content = content.split("```json")[1].split("```")[0].strip()
                                elif "```" in content:
                                    json_content = content.split("```")[1].strip()
                                else:
                                    json_content = content
                                
                                # Parse the updated JSON
                                result_data = json.loads(json_content)
                    
                    # Execute any command blocks
                    command_execution_results = []
                    if self.allow_command_execution and "command_blocks" in result_data:
                        all_command_outputs = []
                        
                        for i, command_block in enumerate(result_data.get("command_blocks", [])):
                            command = command_block.get("command", "")
                            description = command_block.get("description", "")
                            
                            if command:
                                self.logger.info(
                                    message=f"Executing command block {i+1}: {description}",
                                    event_type=LogEventType.COMMAND_EXECUTION,
                                    agent_id=agent_id,
                                    task_id=task_id,
                                    context={
                                        "step_id": step_id,
                                        "command": command
                                    }
                                )
                                
                                # Execute the command
                                execution_result = self.execute_command(command)
                                
                                # Store execution result
                                command_exec_result = {
                                    "command": command,
                                    "description": description,
                                    "success": execution_result.success,
                                    "stdout": execution_result.stdout,
                                    "stderr": execution_result.stderr,
                                    "exit_code": execution_result.exit_code
                                }
                                command_execution_results.append(command_exec_result)
                                
                                # Add to outputs for sharing with LLM if enabled
                                if self.share_output_with_llm:
                                    output_info = f"Command {i+1} ({description}) result:\n"
                                    output_info += f"Command: {command}\n"
                                    output_info += f"Exit code: {execution_result.exit_code}\n"
                                    if execution_result.stdout:
                                        output_info += f"Output:\n```\n{execution_result.stdout}\n```\n"
                                    if execution_result.stderr:
                                        output_info += f"Error:\n```\n{execution_result.stderr}\n```"
                                    all_command_outputs.append(output_info)
                        
                        # If sharing is enabled and we have outputs, send them to the LLM
                        if self.share_output_with_llm and all_command_outputs:
                            command_results_message = "\n\n".join(all_command_outputs)
                            messages.append({"role": "user", "content": f"Here are the results of your command execution:\n\n{command_results_message}\n\nPlease analyze these results and continue with your findings."})
                            
                            # Make another API call with the updated messages and execution results
                            updated_response = await self.service.generate_chat_response(
                                messages=messages,
                                **params
                            )
                            
                            # Update response with the new content that includes analysis of command execution results
                            if "choices" in updated_response and len(updated_response["choices"]) > 0:
                                response = updated_response
                                content = response["choices"][0]["message"]["content"]
                                
                                # Reparse the JSON from the updated response
                                if "```json" in content:
                                    json_content = content.split("```json")[1].split("```")[0].strip()
                                elif "```" in content:
                                    json_content = content.split("```")[1].strip()
                                else:
                                    json_content = content
                                
                                # Parse the updated JSON
                                result_data = json.loads(json_content)
                    
                    # Create StepResult
                    step_result = StepResult(
                        step_id=step.id,
                        success=True,
                        findings=result_data.get("findings", ""),
                        learning=result_data.get("learning", ""),
                        next_steps=result_data.get("next_steps", []),
                        artifacts=result_data.get("artifacts", {}),
                        code_executions=code_execution_results,
                        command_executions=command_execution_results
                    )
                    
                    # Log success
                    self.logger.info(
                        message=f"Step execution completed successfully: {step.title}",
                        event_type=LogEventType.STEP_EXECUTION_COMPLETED,
                        agent_id=agent_id,
                        task_id=task_id,
                        context={
                            "step_id": step_id,
                            "learning": step_result.learning,
                            "findings_snippet": step_result.findings[:100] + "..." if len(step_result.findings) > 100 else step_result.findings,
                            "next_steps_count": len(step_result.next_steps),
                            "code_executions_count": len(code_execution_results),
                            "command_executions_count": len(command_execution_results)
                        }
                    )
                    
                    # Add to execution history
                    self.execution_history.append(step_result)
                    
                    return step_result
                
                except json.JSONDecodeError as e:
                    # If parsing fails, create an error result
                    error_msg = f"JSON decode error: {str(e)}"
                    self.logger.error(
                        message=f"Failed to parse LLM response as JSON: {error_msg}",
                        event_type=LogEventType.ERROR,
                        agent_id=agent_id,
                        task_id=task_id,
                        context={
                            "step_id": step_id,
                            "raw_response_snippet": content[:200] + "..." if len(content) > 200 else content
                        }
                    )
                    
                    error_result = StepResult(
                        step_id=step.id,
                        success=False,
                        findings="Failed to parse LLM response as JSON",
                        learning="Ensure proper JSON formatting in LLM responses",
                        error=error_msg,
                        artifacts={"raw_response": content}
                    )
                    
                    # Add to execution history
                    self.execution_history.append(error_result)
                    
                    return error_result
            
            # Handle case where no valid response was received
            error_msg = "No valid response from LLM"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.ERROR,
                agent_id=agent_id,
                task_id=task_id,
                context={
                    "step_id": step_id,
                    "response_status": str(response)
                }
            )
            
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
            error_msg = f"Error during step execution: {str(e)}"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.ERROR,
                agent_id=agent_id,
                task_id=task_id,
                context={
                    "step_id": step_id,
                    "exception_type": type(e).__name__
                }
            )
            
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