"""
Benchmark evaluation for the MSR framework.
"""
import sys
import os
import argparse
import json
from typing import Dict, List, Any, Optional
import time

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.qwen_adapter import QwenMSRAdapter
from src.reasoning.steps import ReasoningPipeline
from examples.math_problem_solving import create_math_reasoning_pipeline

# Sample benchmark datasets
MATH_PROBLEMS = [
    {
        "id": "math-1",
        "problem": "A train travels at 60 mph. How far will it travel in 2.5 hours?",
        "reference_answer": "150 miles"
    },
    {
        "id": "math-2",
        "problem": "If x + y = 10 and x - y = 4, what is the value of x?",
        "reference_answer": "7"
    },
    {
        "id": "math-3",
        "problem": "A rectangle has a length of 12 cm and a width of 8 cm. What is its area?",
        "reference_answer": "96 square centimeters"
    },
    {
        "id": "math-4",
        "problem": "If a car depreciates in value by 15% per year, what will be its value after 3 years if it originally cost $20,000?",
        "reference_answer": "$12,252.50"
    },
    {
        "id": "math-5",
        "problem": "The sum of three consecutive integers is 72. What are the integers?",
        "reference_answer": "23, 24, and 25"
    }
]

def run_benchmark(pipeline: ReasoningPipeline, problems: List[Dict[str, str]], 
                  output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run benchmark evaluation on a set of problems.
    
    Args:
        pipeline: The reasoning pipeline to evaluate
        problems: List of problem dictionaries with 'id', 'problem', and 'reference_answer'
        output_file: Optional file to write detailed results to
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "total_problems": len(problems),
        "problems_evaluated": 0,
        "total_time": 0,
        "detailed_results": []
    }
    
    for problem_dict in problems:
        problem_id = problem_dict.get("id", "unknown")
        problem = problem_dict.get("problem", "")
        reference = problem_dict.get("reference_answer", "")
        
        print(f"Evaluating problem {problem_id}: {problem}")
        
        # Time the execution
        start_time = time.time()
        
        # Execute the reasoning pipeline
        try:
            result = pipeline.execute(problem)
            
            # Record the result
            execution_time = time.time() - start_time
            
            problem_result = {
                "id": problem_id,
                "problem": problem,
                "reference_answer": reference,
                "prediction": result.get("final_answer", ""),
                "reasoning_steps": result.get("reasoning_steps", []),
                "execution_time": execution_time
            }
            
            results["detailed_results"].append(problem_result)
            results["problems_evaluated"] += 1
            results["total_time"] += execution_time
            
            print(f"  Completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"  Error evaluating problem {problem_id}: {str(e)}")
            
            # Record the error
            problem_result = {
                "id": problem_id,
                "problem": problem,
                "reference_answer": reference,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            results["detailed_results"].append(problem_result)
    
    # Calculate summary statistics
    if results["problems_evaluated"] > 0:
        results["average_time_per_problem"] = results["total_time"] / results["problems_evaluated"]
    else:
        results["average_time_per_problem"] = 0
    
    # Write detailed results to a file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="MSR Framework Benchmark Evaluation")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for detailed benchmark results")
    args = parser.parse_args()
    
    print("Initializing Qwen MSR Adapter...")
    # Initialize the model adapter
    model_adapter = QwenMSRAdapter()
    
    print("Creating math reasoning pipeline...")
    # Create the reasoning pipeline for math problems
    pipeline = create_math_reasoning_pipeline(model_adapter)
    
    print("\nRunning benchmark evaluation...")
    print("-" * 80)
    
    # Run the benchmark
    results = run_benchmark(pipeline, MATH_PROBLEMS, args.output)
    
    # Print summary results
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"Total problems: {results['total_problems']}")
    print(f"Problems evaluated: {results['problems_evaluated']}")
    print(f"Total execution time: {results['total_time']:.2f} seconds")
    print(f"Average time per problem: {results['average_time_per_problem']:.2f} seconds")
    print("-" * 80)
    print(f"Detailed results written to {args.output}")

if __name__ == "__main__":
    main() 