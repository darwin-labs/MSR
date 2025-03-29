#!/usr/bin/env python3
"""
Example of using OpenRouter with the MSR reasoning framework.
"""
import os
import sys
import argparse
import json
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.msr_integration import create_msr_pipeline


def solve_problem(
    problem: str, 
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Solve a problem using the OpenRouter MSR pipeline.
    
    Args:
        problem: Problem to solve
        model: Model to use
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        output_file: File to save detailed results to
        
    Returns:
        Dictionary with reasoning steps and final answer
    """
    # Create MSR pipeline with OpenRouter
    pipeline = create_msr_pipeline(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    print(f"\nSolving problem using {model or 'default model'}:")
    print("-" * 80)
    print(problem)
    print("-" * 80)
    
    # Execute the pipeline on the problem
    print("\nGenerating solution...")
    result = pipeline.execute(problem)
    
    # Print reasoning steps and final answer
    for i, step_info in enumerate(result.get("reasoning_steps", [])):
        step_name = step_info.get("step", f"Step {i+1}")
        step_output = step_info.get("output", "")
        
        print(f"\n## {step_name}:")
        print("-" * 80)
        print(step_output.strip())
        print("-" * 80)
    
    print("\n## Final Answer:")
    print("-" * 80)
    print(result.get("final_answer", "No answer generated."))
    print("-" * 80)
    
    # Save detailed results to a file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    
    return result


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="OpenRouter MSR Reasoning Example")
    parser.add_argument("--model", type=str, 
                        help="Model to use (default: from config or qwen/qwen2.5-32b-instruct)")
    parser.add_argument("--problem", type=str, 
                        default="Solve this math problem step-by-step: If a train travels at 60 mph, how far will it travel in 2.5 hours?",
                        help="Problem to solve")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for generation")
    parser.add_argument("--output", type=str, 
                        help="File to save detailed results to")
    
    args = parser.parse_args()
    
    # Solve the problem
    solve_problem(
        problem=args.problem,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_file=args.output
    )


if __name__ == "__main__":
    main() 