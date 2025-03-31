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

The code will be executed, and the results will be stored and available for subsequent steps.
"""
        
        if self.allow_command_execution:
            capabilities_info += """You can execute terminal commands by including them in your response using the following format:
```execute_command
ls -la
```

The command will be executed, and the results will be stored and available for subsequent steps.
"""
        
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
        prompt = self._create_step_execution_prompt(
            step=step,
            research_plan=research_plan,
            previous_results=previous_results,
            additional_context=additional_context
        )
        
        # Initialize execution records
        code_executions = []
        command_executions = []
        
        # Initialize step result
        step_result = StepResult(
            step_id=step.id,
            success=False,
            findings="",
            learning="",
            code_executions=code_executions,
            command_executions=command_executions
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
                        "prompt_length": len(prompt)
                    }
                )
                
            # Call LLM to execute the step
            response = await self._service.async_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,  # Use model name instead of service object
                temperature=temp,
                max_tokens=tokens,
                **kwargs
            )
            
            # Parse the response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                raise ValueError("Empty response from LLM")
            
            # Extract execution commands
            code_blocks = self._extract_code_blocks(content)
            
            # Execute any Python code or shell commands in the response
            for block in code_blocks:
                block_type = block["type"].lower()
                code = block["code"]
                description = block.get("description", "")
                
                if block_type in ["python", "py"]:
                    # Execute Python code if allowed
                    if self.allow_code_execution:
                        result = self.execute_python_code(code)
                        
                        # Record execution
                        execution_record = {
                            "code": code,
                            "output": result.output,
                            "success": result.success,
                            "error": result.error,
                            "description": description
                        }
                        code_executions.append(execution_record)
                        
                        # Update content with execution result if sharing is enabled
                        if self.share_output_with_llm:
                            content += f"\n\nCode Execution Result:\n```\n{result.output if result.success else result.error}\n```\n"
                            
                elif block_type in ["bash", "shell", "sh", "command"]:
                    # Execute shell command if allowed
                    if self.allow_command_execution:
                        result = self.execute_command(code)
                        
                        # Record execution
                        execution_record = {
                            "command": code,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "exit_code": result.exit_code,
                            "success": result.success,
                            "description": description
                        }
                        command_executions.append(execution_record)
                        
                        # Update content with execution result if sharing is enabled
                        if self.share_output_with_llm:
                            content += f"\n\nCommand Execution Result:\n```\nExit Code: {result.exit_code}\nStdout: {result.stdout}\nStderr: {result.stderr}\n```\n"
            
            # After executing commands, ask LLM to interpret results if there were executions
            if (code_executions or command_executions) and self.share_output_with_llm:
                # Create a prompt with the original response and execution results
                interpretation_prompt = f"Based on the step execution and the results of any code or command executions, what are your findings and learnings from this step? Please provide:\n\n1. Findings: Concrete results or insights\n2. Learning: What we learned\n3. Success: Whether the step was successful (yes/no)\n4. Next Steps: Suggestions for what to do next"
                
                # Call LLM again to interpret results
                interpretation_response = await self._service.async_chat(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": interpretation_prompt}
                    ],
                    model=self.model_name,  # Use model name instead of service object
                    temperature=temp,
                    max_tokens=tokens,
                    **kwargs
                )
                
                # Update content with interpretation
                interpretation_content = interpretation_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if interpretation_content:
                    content += f"\n\n{interpretation_content}"
            
            # Process the complete response
            result = self._parse_step_result(content, step.id)
            
            # Update the step result
            step_result.success = result.get("success", False)
            step_result.findings = result.get("findings", "")
            step_result.learning = result.get("learning", "")
            step_result.next_steps = result.get("next_steps", [])
            
            # Add executions to the result
            step_result.code_executions = code_executions
            step_result.command_executions = command_executions
            
            # Log completion
            if agent_id_for_logs and task_id_for_logs:
                self.logger.info(
                    message=f"Step {step.id} execution completed",
                    agent_id=agent_id_for_logs,
                    task_id=task_id_for_logs,
                    context={
                        "success": step_result.success,
                        "findings_length": len(step_result.findings),
                        "learning_length": len(step_result.learning)
                    }
                )
            
            # Return the result
            return step_result
                
        except Exception as e:
            error_msg = f"Error during step execution: {str(e)}"
            self.logger.error(
                message=error_msg,
                event_type=LogEventType.ERROR
            )
            
            # Update step result with error
            step_result.success = False
            step_result.error = error_msg
            
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