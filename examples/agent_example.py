#!/usr/bin/env python3
"""
Example of using the MSR Agent for task execution with different types of tools.
"""
import os
import sys
import argparse
import json
from typing import Set, Optional
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.msr.agent import Agent, ToolType, create_agent


def run_agent(
    task: str,
    enable_code: bool = False,
    enable_commands: bool = False,
    enable_web_search: bool = False, 
    enable_file_operations: bool = False,
    enable_data_analysis: bool = False,
    enable_visualization: bool = False,
    planner_model: Optional[str] = None,
    executor_model: Optional[str] = None,
    num_steps: int = 5,
    domain: Optional[str] = None,
    additional_context: Optional[str] = None,
    output_file: Optional[str] = None,
    requires_approval: bool = True
):
    """
    Run the agent on a specific task with the specified tools.
    
    Args:
        task: The task to perform
        enable_code: Enable Python code execution
        enable_commands: Enable terminal command execution
        enable_web_search: Enable web search
        enable_file_operations: Enable file read/write operations
        enable_data_analysis: Enable data analysis tools
        enable_visualization: Enable data visualization tools
        planner_model: Model for plan generation
        executor_model: Model for step execution
        num_steps: Suggested number of steps in the plan
        domain: Specific domain for the research
        additional_context: Additional context for the task
        output_file: File to save the results to
        requires_approval: Whether tools require user approval
    """
    print(f"\n{'='*80}")
    print(f"MSR AGENT EXECUTION: {task}")
    print(f"{'='*80}")
    
    # Configure allowed tools
    allowed_tools = set()
    
    if enable_code:
        allowed_tools.add(ToolType.PYTHON_CODE)
    
    if enable_commands:
        allowed_tools.add(ToolType.TERMINAL_COMMAND)
    
    if enable_web_search:
        allowed_tools.add(ToolType.WEB_SEARCH)
    
    if enable_file_operations:
        allowed_tools.add(ToolType.FILE_READ)
        allowed_tools.add(ToolType.FILE_WRITE)
    
    if enable_data_analysis:
        allowed_tools.add(ToolType.DATA_ANALYSIS)
    
    if enable_visualization:
        allowed_tools.add(ToolType.VISUALIZATION)
    
    # Print enabled tools
    print("\nEnabled Tools:")
    for tool_type in allowed_tools:
        print(f"- {tool_type.name}")
    
    # Create a temporary file for state saving
    state_file = "agent_state.json"
    
    # Create agent
    agent = create_agent(
        task=task,
        allowed_tools=allowed_tools or None,
        planner_model=planner_model,
        executor_model=executor_model,
        save_state_path=state_file
    )
    
    # Set approval requirement
    agent.requires_tool_approval = requires_approval
    
    # Run the agent
    print(f"\nGenerating and executing plan with {num_steps} suggested steps...")
    
    try:
        result = agent.run(
            num_steps=num_steps,
            domain=domain,
            additional_context=additional_context
        )
        
        # Print results summary
        print(f"\n{'='*80}")
        print("EXECUTION RESULTS")
        print(f"{'='*80}")
        
        print(f"\nPlan: {result['plan']['title']}")
        print(f"Steps Completed: {result['summary']['successful_steps']}/{result['summary']['total_steps']}")
        print(f"Success Rate: {result['success_rate']*100:.1f}%")
        
        print("\nKey Learnings:")
        print(result['summary']['key_learnings'])
        
        # Save results if requested
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return result
        
    finally:
        # Clean up temporary state file
        if os.path.exists(state_file):
            os.remove(state_file)


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="MSR Agent Example")
    
    parser.add_argument("--task", type=str, required=True,
                       help="The task for the agent to perform")
    
    # Tool configuration
    tool_group = parser.add_argument_group("Tool Configuration")
    tool_group.add_argument("--enable-code", action="store_true",
                          help="Enable Python code execution")
    tool_group.add_argument("--enable-commands", action="store_true", 
                          help="Enable terminal command execution")
    tool_group.add_argument("--enable-web", action="store_true",
                          help="Enable web search")
    tool_group.add_argument("--enable-files", action="store_true",
                          help="Enable file operations")
    tool_group.add_argument("--enable-data", action="store_true",
                          help="Enable data analysis tools")
    tool_group.add_argument("--enable-viz", action="store_true",
                          help="Enable data visualization tools")
    tool_group.add_argument("--enable-all", action="store_true",
                          help="Enable all tools")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--planner-model", type=str,
                           help="Model to use for planning")
    model_group.add_argument("--executor-model", type=str,
                           help="Model to use for step execution")
    
    # Plan configuration
    plan_group = parser.add_argument_group("Plan Configuration")
    plan_group.add_argument("--steps", type=int, default=5,
                          help="Suggested number of steps (default: 5)")
    plan_group.add_argument("--domain", type=str,
                          help="Specific domain for the research")
    plan_group.add_argument("--context", type=str,
                          help="Additional context for the task")
    
    # Output configuration
    parser.add_argument("--output", type=str,
                       help="File to save results to")
    
    # Execution configuration
    parser.add_argument("--no-approval", action="store_true",
                       help="Don't require approval for tool execution")
    
    args = parser.parse_args()
    
    # Process enable-all flag
    if args.enable_all:
        args.enable_code = True
        args.enable_commands = True
        args.enable_web = True
        args.enable_files = True
        args.enable_data = True
        args.enable_viz = True
    
    # Run the agent
    run_agent(
        task=args.task,
        enable_code=args.enable_code,
        enable_commands=args.enable_commands,
        enable_web_search=args.enable_web,
        enable_file_operations=args.enable_files,
        enable_data_analysis=args.enable_data,
        enable_visualization=args.enable_viz,
        planner_model=args.planner_model,
        executor_model=args.executor_model,
        num_steps=args.steps,
        domain=args.domain,
        additional_context=args.context,
        output_file=args.output,
        requires_approval=not args.no_approval
    )


if __name__ == "__main__":
    main() 