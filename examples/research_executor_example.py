#!/usr/bin/env python3
"""
Example of using the PlannerLLM and StepExecutor to generate and execute research plans.
"""
import os
import sys
import argparse
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.msr.planner_llm import create_planner, ResearchPlan
from src.msr.step_executor import create_step_executor, StepResult


def run_research_workflow(
    prompt: str,
    model: str = None,
    executor_model: str = None,
    num_steps: int = 5,
    domain: str = None,
    temperature: float = 0.2,
    executor_temperature: float = 0.7,
    output_file: str = None,
    additional_context: str = None
) -> List[StepResult]:
    """
    Generate and execute a research plan.
    
    Args:
        prompt: Research prompt or question
        model: Planner model to use (default: Claude 3 Opus)
        executor_model: Executor model to use (default: Claude 3 Opus)
        num_steps: Suggested number of steps (default: 5)
        domain: Research domain (optional)
        temperature: Temperature for plan generation (default: 0.2)
        executor_temperature: Temperature for step execution (default: 0.7)
        output_file: File to save the results to (optional)
        additional_context: Additional context for execution (optional)
        
    Returns:
        List of step execution results
    """
    print(f"\n{'='*80}")
    print(f"RESEARCH WORKFLOW: {prompt}")
    print(f"{'='*80}")
    
    # PHASE 1: Generate research plan
    print("\n[PHASE 1] Generating research plan...")
    
    # Create planner with the specified model
    planner = create_planner(model=model, temperature=temperature)
    
    print(f"Model: {model or 'Default (Claude 3 Opus)'}")
    print(f"Suggested number of steps: {num_steps} (will vary based on task complexity)")
    if domain:
        print(f"Domain: {domain}")
    
    # Generate the plan
    plan = planner.generate_plan(
        prompt=prompt,
        num_steps=num_steps,
        domain=domain
    )
    
    # Print the plan summary
    print(f"\nGenerated Research Plan: {plan.title}")
    print(f"Objective: {plan.objective}")
    print(f"Description: {plan.description}")
    print(f"Number of steps: {len(plan.steps)}")
    
    # PHASE 2: Execute research steps
    print(f"\n{'='*80}")
    print("[PHASE 2] Executing research steps...")
    print(f"{'='*80}")
    
    # Create executor
    executor = create_step_executor(
        model=executor_model,
        temperature=executor_temperature
    )
    
    # Execute all steps in the plan
    results = executor.execute_plan(
        research_plan=plan,
        additional_context=additional_context
    )
    
    # Print execution results
    print(f"\nRESEARCH EXECUTION RESULTS")
    print(f"{'='*80}")
    
    successful_steps = len([r for r in results if r.success])
    print(f"Completed {successful_steps}/{len(results)} steps successfully")
    
    for i, result in enumerate(results, 1):
        step_id = result.step_id
        step = next((s for s in plan.steps if s.id == step_id), None)
        
        print(f"\n{i}. Step: {step.title if step else step_id}")
        print(f"   Status: {'✓ Success' if result.success else '✗ Failed'}")
        
        if result.success:
            print(f"   Key Learning: {result.learning}")
            print(f"   Findings Summary: {result.findings[:200]}...")
            
            if result.next_steps:
                print(f"   Suggested Next Steps: {', '.join(result.next_steps[:3])}")
                
            if result.artifacts and 'references' in result.artifacts:
                refs = result.artifacts['references']
                if refs and isinstance(refs, list):
                    print(f"   References: {', '.join(refs[:3])}")
        else:
            print(f"   Error: {result.error}")
    
    # Save to file if requested
    if output_file:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Prepare output data
        output_data = {
            "research_plan": plan.to_dict(),
            "execution_results": [r.to_dict() for r in results]
        }
        
        # Save as JSON
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResearch results saved to: {output_file}")
    
    # Return all step results
    return results


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Research Workflow Example")
    parser.add_argument("--prompt", type=str, 
                        default="What are the most effective methods for reducing carbon emissions in urban areas?",
                        help="Research prompt or question")
    parser.add_argument("--model", type=str, 
                        help="Planner model to use (default: anthropic/claude-3-opus)")
    parser.add_argument("--executor-model", type=str, 
                        help="Executor model to use (default: same as planner)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Suggested number of steps (LLM will adjust based on task complexity)")
    parser.add_argument("--domain", type=str,
                        help="Research domain (e.g., 'climate science', 'AI')")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for plan generation (default: 0.2)")
    parser.add_argument("--executor-temperature", type=float, default=0.7,
                        help="Temperature for step execution (default: 0.7)")
    parser.add_argument("--context", type=str,
                        help="Additional context for research execution")
    parser.add_argument("--output", type=str,
                        help="File to save the research results to (as JSON)")
    
    args = parser.parse_args()
    
    # Run the research workflow
    run_research_workflow(
        prompt=args.prompt,
        model=args.model,
        executor_model=args.executor_model,
        num_steps=args.steps,
        domain=args.domain,
        temperature=args.temperature,
        executor_temperature=args.executor_temperature,
        additional_context=args.context,
        output_file=args.output
    )


if __name__ == "__main__":
    main() 