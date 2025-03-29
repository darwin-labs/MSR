"""
Tests for the reasoning steps module.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reasoning.steps import (
    ReasoningStep, 
    PlanningStep, 
    DecompositionStep, 
    SolutionStep, 
    VerificationStep, 
    ReasoningPipeline
)

class TestReasoningSteps(unittest.TestCase):
    """Test case for reasoning steps implementation."""
    
    def setUp(self):
        """Set up for tests."""
        # Create a mock model function for testing
        self.mock_model_fn = MagicMock()
        self.mock_model_fn.return_value = "Mock model output"
        
        # Create a test problem
        self.test_problem = "What is 2 + 2?"
        self.test_context = {"problem": self.test_problem}
    
    def test_planning_step(self):
        """Test the planning step."""
        planning_step = PlanningStep(self.mock_model_fn)
        
        # Execute the step
        result = planning_step.execute(self.test_context)
        
        # Check that the model function was called
        self.mock_model_fn.assert_called_once()
        
        # Check that the result contains the expected keys
        self.assertIn("plan", result)
        self.assertIn("reasoning_steps", result)
        
        # Check that the planning output was added to reasoning steps
        self.assertEqual(len(result["reasoning_steps"]), 1)
        self.assertEqual(result["reasoning_steps"][0]["step"], "Planning")
    
    def test_decomposition_step(self):
        """Test the decomposition step."""
        decomposition_step = DecompositionStep(self.mock_model_fn)
        
        # Add a plan to the context
        context = dict(self.test_context)
        context["plan"] = "Test plan"
        
        # Execute the step
        result = decomposition_step.execute(context)
        
        # Check that the model function was called
        self.mock_model_fn.assert_called_once()
        
        # Check that the result contains the expected keys
        self.assertIn("decomposition", result)
        self.assertIn("subproblems", result)
        self.assertIn("reasoning_steps", result)
        
        # Check that the decomposition output was added to reasoning steps
        self.assertEqual(len(result["reasoning_steps"]), 1)
        self.assertEqual(result["reasoning_steps"][0]["step"], "Decomposition")
    
    def test_solution_step(self):
        """Test the solution step."""
        solution_step = SolutionStep(self.mock_model_fn)
        
        # Add a plan and decomposition to the context
        context = dict(self.test_context)
        context["plan"] = "Test plan"
        context["decomposition"] = "Test decomposition"
        
        # Execute the step
        result = solution_step.execute(context)
        
        # Check that the model function was called
        self.mock_model_fn.assert_called_once()
        
        # Check that the result contains the expected keys
        self.assertIn("solution", result)
        self.assertIn("reasoning_steps", result)
        
        # Check that the solution output was added to reasoning steps
        self.assertEqual(len(result["reasoning_steps"]), 1)
        self.assertEqual(result["reasoning_steps"][0]["step"], "Solution")
    
    def test_verification_step(self):
        """Test the verification step."""
        verification_step = VerificationStep(self.mock_model_fn)
        
        # Add a solution to the context
        context = dict(self.test_context)
        context["solution"] = "Test solution"
        
        # Execute the step
        result = verification_step.execute(context)
        
        # Check that the model function was called
        self.mock_model_fn.assert_called_once()
        
        # Check that the result contains the expected keys
        self.assertIn("verification", result)
        self.assertIn("reasoning_steps", result)
        
        # Check that the verification output was added to reasoning steps
        self.assertEqual(len(result["reasoning_steps"]), 1)
        self.assertEqual(result["reasoning_steps"][0]["step"], "Verification")
    
    def test_reasoning_pipeline(self):
        """Test the full reasoning pipeline."""
        # Create a pipeline with all steps
        pipeline = ReasoningPipeline([
            PlanningStep(self.mock_model_fn),
            DecompositionStep(self.mock_model_fn),
            SolutionStep(self.mock_model_fn),
            VerificationStep(self.mock_model_fn)
        ])
        
        # Execute the pipeline
        result = pipeline.execute(self.test_problem)
        
        # Check that the model function was called 4 times (once per step)
        self.assertEqual(self.mock_model_fn.call_count, 4)
        
        # Check that the result contains the expected keys
        self.assertIn("problem", result)
        self.assertIn("plan", result)
        self.assertIn("decomposition", result)
        self.assertIn("solution", result)
        self.assertIn("verification", result)
        self.assertIn("final_answer", result)
        self.assertIn("reasoning_steps", result)
        
        # Check that all steps were added to reasoning steps
        self.assertEqual(len(result["reasoning_steps"]), 4)
        self.assertEqual(result["reasoning_steps"][0]["step"], "Planning")
        self.assertEqual(result["reasoning_steps"][1]["step"], "Decomposition")
        self.assertEqual(result["reasoning_steps"][2]["step"], "Solution")
        self.assertEqual(result["reasoning_steps"][3]["step"], "Verification")

if __name__ == "__main__":
    unittest.main() 