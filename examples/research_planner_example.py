#!/usr/bin/env python3
"""
Example of using the PlannerLLM to generate structured research plans.
"""
import os
import sys
import argparse
import json
from typing import Dict, Any
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.msr.planner_llm import create_planner, ResearchPlan


def generate_plan(
    prompt: str,
    model: str = None,
    num_steps: int = 5,
    domain: str = None,
    temperature: float = 0.2,
    output_file: str = None
) -> ResearchPlan:
    """
    Generate a research plan using PlannerLLM.
    
    Args:
        prompt: Research prompt or question
        model: Model to use (default: Claude 3 Opus)
        num_steps: Number of steps in the plan (default: 5)
        domain: Research domain (optional)
        temperature: Temperature for generation (default: 0.2)
        output_file: File to save the plan to (optional)
        
    Returns:
        Generated research plan
    """
    # Create planner with the specified model
    planner = create_planner(model=model, temperature=temperature)
    
    print(f"\nGenerating research plan for prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print(f"Model: {model or 'Default (Claude 3 Opus)'}")
    print(f"Number of steps: {num_steps}")
    if domain:
        print(f"Domain: {domain}")
    print()
    
    # Generate the plan
    plan = planner.generate_plan(
        prompt=prompt,
        num_steps=num_steps,
        domain=domain
    )
    
    # Print the plan summary
    print("\nGenerated Research Plan:")
    print("=" * 80)
    print(f"Title: {plan.title}")
    print(f"Objective: {plan.objective}")
    print("-" * 80)
    print(f"Description: {plan.description}")
    print("-" * 80)
    
    # Print the steps
    print("\nResearch Steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"\n{i}. {step.title} (ID: {step.id})")
        if step.dependencies:
            print(f"   Dependencies: {', '.join(step.dependencies)}")
        print(f"   Goal: {step.goal}")
        print(f"   Description: {step.description}")
        print(f"   Expected output: {step.expected_output}")
    
    # Print metadata if available
    if plan.metadata:
        print("\nMetadata:")
        print("-" * 80)
        for key, value in plan.metadata.items():
            print(f"{key}: {value}")
    
    # Save to file if requested
    if output_file:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save as JSON
        with open(output_file, "w") as f:
            f.write(plan.to_json())
        print(f"\nResearch plan saved to: {output_file}")
    
    return plan


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Research Planner Example")
    parser.add_argument("--prompt", type=str, 
                        default="What are the most effective methods for reducing carbon emissions in urban areas?",
                        help="Research prompt or question")
    parser.add_argument("--model", type=str, 
                        help="Model to use (default: anthropic/claude-3-opus)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of steps in the research plan")
    parser.add_argument("--domain", type=str,
                        help="Research domain (e.g., 'climate science', 'AI')")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation (default: 0.2)")
    parser.add_argument("--output", type=str,
                        help="File to save the research plan to (as JSON)")
    
    args = parser.parse_args()
    
    # Generate research plan
    generate_plan(
        prompt=args.prompt,
        model=args.model,
        num_steps=args.steps,
        domain=args.domain,
        temperature=args.temperature,
        output_file=args.output
    )


if __name__ == "__main__":
    main() 