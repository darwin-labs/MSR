"""
Adapter for Qwen QWQ 32B model integration with MSR framework.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union

class QwenMSRAdapter:
    """
    Adapter class for Qwen QWQ 32B to work with the MSR framework.
    
    This class handles the loading, configuration, and inference patterns
    specific to using Qwen models with multi-step reasoning.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-32B-Instruct", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Initialize the Qwen adapter.
        
        Args:
            model_name: The specific Qwen model to load
            device: The device to load the model on
            **kwargs: Additional arguments passed to the model loading function
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=device,
            **kwargs
        )
        
    def generate_reasoning_steps(
        self, 
        prompt: str, 
        num_steps: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Generate explicit reasoning steps for a given prompt.
        
        Args:
            prompt: The input problem or question
            num_steps: Number of reasoning steps to generate
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate per step
            **kwargs: Additional generation parameters
            
        Returns:
            A list of reasoning step outputs
        """
        reasoning_steps = []
        current_context = prompt
        
        for step in range(num_steps):
            step_prompt = f"{current_context}\n\nStep {step+1}: "
            inputs = self.tokenizer(step_prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            
            step_output = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            reasoning_steps.append(step_output)
            current_context = f"{current_context}\n\nStep {step+1}: {step_output}"
        
        return reasoning_steps
    
    def generate_with_reasoning(
        self, 
        prompt: str,
        temperature: float = 0.7,
        num_reasoning_steps: int = 3,
        max_new_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Generate a response with explicit multi-step reasoning.
        
        Args:
            prompt: The input problem or question
            temperature: Sampling temperature
            num_reasoning_steps: Number of reasoning steps to generate
            max_new_tokens: Maximum number of tokens to generate for final answer
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing reasoning steps and final answer
        """
        # Generate reasoning steps
        reasoning_steps = self.generate_reasoning_steps(
            prompt=prompt,
            num_steps=num_reasoning_steps,
            temperature=temperature,
            **kwargs
        )
        
        # Build full context with reasoning
        full_context = prompt
        for i, step in enumerate(reasoning_steps):
            full_context += f"\n\nStep {i+1}: {step}"
        
        full_context += "\n\nFinal Answer: "
        
        # Generate final answer
        inputs = self.tokenizer(full_context, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        final_answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer
        } 