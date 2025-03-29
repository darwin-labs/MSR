"""
Example application of MSR framework on mathematical problem solving.
"""
import sys
import os
import argparse

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.qwen_adapter import QwenMSRAdapter
from src.reasoning.steps import (
    ReasoningPipeline, 
    PlanningStep, 
    DecompositionStep, 
    SolutionStep, 
    VerificationStep
)

def create_math_reasoning_pipeline(model_adapter):
    """
    Create a reasoning pipeline for solving math problems.
    
    Args:
        model_adapter: The model adapter to use for generating steps
        
    Returns:
        A ReasoningPipeline configured for math problem solving
    """
    # Create a simple wrapper for the model adapter's generation function
    def model_fn(prompt, **kwargs):
        # Use reasonable defaults for math problems
        default_kwargs = {
            "temperature": 0.3,
            "max_new_tokens": 512
        }
        # Override defaults with any provided kwargs
        for k, v in kwargs.items():
            default_kwargs[k] = v
            
        # Call the model to generate reasoning with our prompt
        response = model_adapter.generate_with_reasoning(
            prompt=prompt, 
            num_reasoning_steps=1,  # We control steps explicitly in our pipeline
            **default_kwargs
        )
        
        # Return just the final answer component
        return response.get("final_answer", "")
    
    # Create the reasoning pipeline with appropriate steps
    pipeline = ReasoningPipeline()
    
    # Add reasoning steps for math problem solving
    pipeline.add_step(PlanningStep(model_fn))
    pipeline.add_step(DecompositionStep(model_fn))
    pipeline.add_step(SolutionStep(model_fn))
    pipeline.add_step(VerificationStep(model_fn))
    
    return pipeline

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="MSR Framework Math Problem Solving Example")
    parser.add_argument("--problem", type=str, 
                        default="A train travels at 60 mph. How far will it travel in 2.5 hours?",
                        help="The math problem to solve")
    args = parser.parse_args()
    
    print("Initializing Qwen MSR Adapter...")
    # Initialize the model adapter
    # Note: In a real application, you might want to add parameters for model selection
    model_adapter = QwenMSRAdapter()
    
    print("Creating math reasoning pipeline...")
    # Create the reasoning pipeline
    pipeline = create_math_reasoning_pipeline(model_adapter)
    
    print(f"\nSolving problem: {args.problem}\n")
    print("-" * 80)
    
    # Execute the reasoning pipeline on the problem
    result = pipeline.execute(args.problem)
    
    # Print the reasoning steps and final answer
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

if __name__ == "__main__":
    main() 