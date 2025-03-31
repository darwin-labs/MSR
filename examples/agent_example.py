#!/usr/bin/env python3
"""
Example of using the MSR Agent for task execution with different types of tools.
"""
import os
import sys
import argparse
import json
import asyncio
from typing import Set, Optional, Dict, List, Any
from pathlib import Path
import logging

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.msr.agent import Agent, ToolType, create_agent
from src.msr.planner_llm import PlannerLLM
from src.msr.logger import configure_logger, LogLevel


async def run_agent(
    task: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    steps: int = 3,
    output_file: Optional[str] = None,
    approve_execution: bool = True,
    enable_python: bool = False,
    enable_terminal: bool = False,
    enable_web_search: bool = False,
    enable_file_operations: bool = False,
    enable_data_analysis: bool = False,
    enable_visualization: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Run an agent with the specified configuration.
    
    Args:
        task: The task description for the agent
        model: The model to use (defaults to config or Claude 3 Opus)
        temperature: The temperature for generation (creativity)
        steps: Maximum number of steps in the plan
        output_file: Optional file path to save the agent state as JSON
        approve_execution: Whether to require approval before each step
        enable_python: Whether to enable Python code execution
        enable_terminal: Whether to enable terminal commands
        enable_web_search: Whether to enable web search
        enable_file_operations: Whether to enable file operations
        enable_data_analysis: Whether to enable data analysis
        enable_visualization: Whether to enable data visualization
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
    """
    # Configure logger
    log_level_enum = LogLevel[log_level.upper()]
    configure_logger(
        level=log_level_enum,
        log_file=log_file,
        log_format="json" if log_file else "text"
    )
    
    print(f"🤖 Running agent with task: {task}")
    print(f"📝 Using model: {model or 'default'}, Temperature: {temperature}")
    
    # Determine which tools to enable
    allowed_tools = set()
    
    if enable_python:
        allowed_tools.add(ToolType.PYTHON_EXECUTION)
        print("🐍 Enabled Python code execution")
        
    if enable_terminal:
        allowed_tools.add(ToolType.TERMINAL_COMMAND)
        print("💻 Enabled terminal command execution")
        
    if enable_web_search:
        allowed_tools.add(ToolType.WEB_SEARCH)
        print("🔍 Enabled web search")
        
    if enable_file_operations:
        allowed_tools.add(ToolType.FILE_OPERATIONS)
        print("📂 Enabled file operations")
        
    if enable_data_analysis:
        allowed_tools.add(ToolType.DATA_ANALYSIS)
        print("📊 Enabled data analysis")
        
    if enable_visualization:
        allowed_tools.add(ToolType.VISUALIZATION)
        print("📈 Enabled data visualization")
    
    # Create agent
    agent = create_agent(
        task=task,
        allowed_tools=allowed_tools,
        model=model,
        temperature=temperature,
        require_step_approval=approve_execution
    )
    
    # Configure planner
    planner_config = {
        "model": model,
        "temperature": temperature,
        "max_steps": steps
    }
    
    # Run the agent
    print("\n🚀 Starting agent execution...\n")
    await agent.run(planner_config=planner_config)
    
    # If output file is specified, save state to file
    if output_file:
        print(f"\n💾 Saving agent state to {output_file}")
        agent.save_state(output_file)
    
    # Print completion message
    print("\n✅ Agent execution complete!")
    print(f"📊 Statistics: {len([s for s in agent.executed_steps if s.success])}/{len(agent.executed_steps)} steps successful")
    
    # Print where to find logs if log_file was specified
    if log_file:
        print(f"\n📋 Detailed logs saved to: {log_file}")
        print("   You can view structured logs with: jq '.' " + log_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MSR Agent with specified tools and configuration")
    parser.add_argument("task", help="The task to execute")
    parser.add_argument("--model", help="Model to use (e.g., qwen/qwen2.5-32b-instruct)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model sampling")
    parser.add_argument("--steps", type=int, default=5, help="Maximum number of steps in the plan")
    parser.add_argument("--enable-python", action="store_true", help="Enable Python code execution")
    parser.add_argument("--enable-terminal", action="store_true", help="Enable terminal command execution")
    parser.add_argument("--enable-web-search", action="store_true", help="Enable web search")
    parser.add_argument("--enable-file-operations", action="store_true", help="Enable file operations")
    parser.add_argument("--enable-data-analysis", action="store_true", help="Enable data analysis")
    parser.add_argument("--enable-visualization", action="store_true", help="Enable visualization")
    parser.add_argument("--output-file", help="File to save the output to")
    parser.add_argument("--no-approval", action="store_true", help="Skip step approval")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    parser.add_argument("--log-file", default="logs/agent.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Run the agent
    run_agent(
        task=args.task,
        model=args.model,
        temperature=args.temperature,
        steps=args.steps,
        enable_python=args.enable_python,
        enable_terminal=args.enable_terminal,
        enable_web_search=args.enable_web_search,
        enable_file_operations=args.enable_file_operations,
        enable_data_analysis=args.enable_data_analysis,
        enable_visualization=args.enable_visualization,
        output_file=args.output_file,
        approve_execution=not args.no_approval
    )


if __name__ == "__main__":
    main() 