"""
MSR Logger module for tracking agent activities across the framework.
"""
import os
import sys
import logging
import json
import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum, auto
from pathlib import Path


class LogLevel(Enum):
    """Log levels for the MSR Logger."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class LogEventType(Enum):
    """Types of events that can be logged."""
    AGENT_INITIALIZED = auto()
    AGENT_TASK_STARTED = auto()
    AGENT_TASK_COMPLETED = auto()
    PLAN_GENERATION_STARTED = auto()
    PLAN_GENERATION_COMPLETED = auto()
    STEP_EXECUTION_STARTED = auto()
    STEP_EXECUTION_COMPLETED = auto()
    TOOL_EXECUTION_STARTED = auto()
    TOOL_EXECUTION_COMPLETED = auto()
    CODE_EXECUTION = auto()
    COMMAND_EXECUTION = auto()
    ERROR = auto()
    WARNING = auto()
    STATE_SAVED = auto()
    STATE_LOADED = auto()


class MSRLogger:
    """
    Logger for MSR framework that tracks agent activities.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True,
        log_level: LogLevel = LogLevel.INFO,
        structured_logging: bool = True,
        include_timestamp: bool = True
    ):
        """
        Initialize the MSR Logger.
        
        Args:
            log_file: Path to log file (optional)
            console_output: Whether to output logs to console
            log_level: Minimum log level to record
            structured_logging: Whether to use structured JSON logging
            include_timestamp: Whether to include timestamps in logs
        """
        self.log_file = log_file
        self.console_output = console_output
        self.log_level = log_level
        self.structured_logging = structured_logging
        self.include_timestamp = include_timestamp
        
        # Initialize Python logger
        self.logger = logging.getLogger("msr")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Configure console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_python_log_level(log_level))
            
            # Create formatter
            if structured_logging:
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Configure file handler if requested
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self._get_python_log_level(log_level))
            
            # Create formatter
            if structured_logging:
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _get_python_log_level(self, level: LogLevel) -> int:
        """Convert MSR LogLevel to Python logging level."""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return level_map.get(level, logging.INFO)
    
    def _format_log_message(
        self,
        event_type: LogEventType,
        message: str,
        level: LogLevel,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> str:
        """Format log message as structured JSON or plain text."""
        if self.structured_logging:
            log_data = {
                "event_type": event_type.name,
                "message": message,
                "level": level.name
            }
            
            if agent_id:
                log_data["agent_id"] = agent_id
                
            if task_id:
                log_data["task_id"] = task_id
                
            if context:
                log_data["context"] = context
                
            if self.include_timestamp:
                log_data["timestamp"] = datetime.datetime.now().isoformat()
                
            return json.dumps(log_data)
        else:
            # Simple text format
            prefix = ""
            if agent_id:
                prefix += f"[Agent: {agent_id}] "
            if task_id:
                prefix += f"[Task: {task_id}] "
                
            return f"{prefix}[{event_type.name}] {message}"
    
    def _log(
        self,
        event_type: LogEventType,
        message: str,
        level: LogLevel,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Internal method to log messages at the specified level."""
        # Skip if below minimum log level
        if level.value < self.log_level.value:
            return
        
        # Format the message
        formatted_message = self._format_log_message(
            event_type=event_type,
            message=message,
            level=level,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
        
        # Log with the appropriate level
        if level == LogLevel.DEBUG:
            self.logger.debug(formatted_message)
        elif level == LogLevel.INFO:
            self.logger.info(formatted_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(formatted_message)
        elif level == LogLevel.ERROR:
            self.logger.error(formatted_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(formatted_message)
    
    def debug(
        self,
        message: str,
        event_type: LogEventType = LogEventType.AGENT_TASK_STARTED,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Log a debug message."""
        self._log(
            event_type=event_type,
            message=message,
            level=LogLevel.DEBUG,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def info(
        self,
        message: str,
        event_type: LogEventType = LogEventType.AGENT_TASK_STARTED,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Log an info message."""
        self._log(
            event_type=event_type,
            message=message,
            level=LogLevel.INFO,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def warning(
        self,
        message: str,
        event_type: LogEventType = LogEventType.WARNING,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Log a warning message."""
        self._log(
            event_type=event_type,
            message=message,
            level=LogLevel.WARNING,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def error(
        self,
        message: str,
        event_type: LogEventType = LogEventType.ERROR,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Log an error message."""
        self._log(
            event_type=event_type,
            message=message,
            level=LogLevel.ERROR,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def critical(
        self,
        message: str,
        event_type: LogEventType = LogEventType.ERROR,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Log a critical message."""
        self._log(
            event_type=event_type,
            message=message,
            level=LogLevel.CRITICAL,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_agent_initialized(
        self, 
        agent_id: str,
        task: str,
        allowed_tools: List[str],
        context: Optional[Dict[str, Any]] = None
    ):
        """Log agent initialization."""
        self.info(
            message=f"Agent initialized with task: {task}",
            event_type=LogEventType.AGENT_INITIALIZED,
            context={
                "allowed_tools": allowed_tools,
                **(context or {})
            },
            agent_id=agent_id
        )
    
    def log_task_started(
        self,
        agent_id: str,
        task_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log task started."""
        self.info(
            message=f"Task started: {task_description}",
            event_type=LogEventType.AGENT_TASK_STARTED,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_task_completed(
        self,
        agent_id: str,
        task_id: str,
        success: bool,
        result_summary: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log task completion."""
        self.info(
            message=f"Task {'completed successfully' if success else 'failed'}",
            event_type=LogEventType.AGENT_TASK_COMPLETED,
            context={
                "success": success,
                "result_summary": result_summary,
                **(context or {})
            },
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_plan_generation_started(
        self,
        agent_id: str,
        task_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log plan generation started."""
        self.info(
            message="Plan generation started",
            event_type=LogEventType.PLAN_GENERATION_STARTED,
            context={
                "prompt": prompt,
                **(context or {})
            },
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_plan_generation_completed(
        self,
        agent_id: str,
        task_id: str,
        plan_title: str,
        num_steps: int,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log plan generation completed."""
        self.info(
            message=f"Plan generation completed: {plan_title} with {num_steps} steps",
            event_type=LogEventType.PLAN_GENERATION_COMPLETED,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_step_execution_started(
        self,
        agent_id: str,
        task_id: str,
        step_id: str,
        step_title: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log step execution started."""
        self.info(
            message=f"Step execution started: {step_title}",
            event_type=LogEventType.STEP_EXECUTION_STARTED,
            context=context,
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_step_execution_completed(
        self,
        agent_id: str,
        task_id: str,
        step_id: str,
        success: bool,
        findings: Optional[str] = None,
        learning: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log step execution completed."""
        self.info(
            message=f"Step {step_id} execution {'completed successfully' if success else 'failed'}",
            event_type=LogEventType.STEP_EXECUTION_COMPLETED,
            context={
                "success": success,
                "findings": findings,
                "learning": learning,
                **(context or {})
            },
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_tool_execution(
        self,
        agent_id: str,
        task_id: str,
        tool_name: str,
        success: bool,
        input_data: Any = None,
        output_data: Any = None,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log tool execution."""
        event_type = LogEventType.TOOL_EXECUTION_COMPLETED
        
        if tool_name.lower() == "execute_python":
            event_type = LogEventType.CODE_EXECUTION
        elif tool_name.lower() == "execute_command":
            event_type = LogEventType.COMMAND_EXECUTION
        
        self.info(
            message=f"Tool execution: {tool_name} {'succeeded' if success else 'failed'}",
            event_type=event_type,
            context={
                "tool_name": tool_name,
                "success": success,
                "input": input_data,
                "output": output_data,
                "error": error,
                **(context or {})
            },
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_state_saved(
        self,
        agent_id: str,
        path: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log agent state saved."""
        self.info(
            message=f"Agent state saved to {path}",
            event_type=LogEventType.STATE_SAVED,
            context=context,
            agent_id=agent_id
        )
    
    def log_state_loaded(
        self,
        agent_id: str,
        path: str,
        success: bool,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log agent state loaded."""
        self.info(
            message=f"Agent state {'loaded from' if success else 'failed to load from'} {path}",
            event_type=LogEventType.STATE_LOADED,
            context={
                "success": success,
                "error": error,
                **(context or {})
            },
            agent_id=agent_id
        )


# Global logger instance
_default_logger = MSRLogger()

def get_logger() -> MSRLogger:
    """Get the default logger instance."""
    return _default_logger

def configure_logger(
    log_file: Optional[str] = None,
    console_output: bool = True,
    log_level: LogLevel = LogLevel.INFO,
    structured_logging: bool = True,
    include_timestamp: bool = True
) -> MSRLogger:
    """
    Configure the global logger.
    
    Args:
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console
        log_level: Minimum log level to record
        structured_logging: Whether to use structured JSON logging
        include_timestamp: Whether to include timestamps in logs
        
    Returns:
        Configured logger instance
    """
    global _default_logger
    _default_logger = MSRLogger(
        log_file=log_file,
        console_output=console_output,
        log_level=log_level,
        structured_logging=structured_logging,
        include_timestamp=include_timestamp
    )
    return _default_logger 