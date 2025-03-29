#!/usr/bin/env python3
"""
Example of using the OpenRouter service in the MSR framework.
"""
import os
import sys
import argparse
import asyncio
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_service.openrouter_service import create_openrouter_service


async def list_models():
    """List all available models from OpenRouter."""
    # Create service
    service = create_openrouter_service()
    
    try:
        # Get models
        models_data = await service.get_available_models_async()
        
        # Print model information
        print("\nAvailable models from OpenRouter:")
        print("-" * 80)
        
        for model in models_data:
            model_id = model.get("id", "Unknown")
            context_length = model.get("context_length", "Unknown")
            pricing = model.get("pricing", {})
            
            # Format pricing info if available
            pricing_info = ""
            if pricing:
                input_price = pricing.get("input", 0)
                output_price = pricing.get("output", 0)
                pricing_info = f"(Input: ${input_price}/1M tokens, Output: ${output_price}/1M tokens)"
            
            print(f"- {model_id}")
            print(f"  Context length: {context_length} tokens")
            if pricing_info:
                print(f"  Pricing: {pricing_info}")
            print()
    
    finally:
        # Clean up
        await service._close_session()


async def chat_example(model: str, prompt: str, temperature: float, max_tokens: int):
    """Run a chat example with OpenRouter."""
    # Create service
    service = create_openrouter_service()
    
    try:
        print(f"\nSending prompt to {model}:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
        
        # Set up chat messages
        messages = [{"role": "user", "content": prompt}]
        
        # Get response
        print("\nGenerating response...")
        response = await service.generate_chat_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model
        )
        
        # Extract and print content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            
            print("\nResponse:")
            print("-" * 80)
            print(content)
            print("-" * 80)
            
            # Print model and usage information
            print("\nModel Information:")
            print(f"Model: {response.get('model', 'Unknown')}")
            
            usage = response.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", "Unknown")
                completion_tokens = usage.get("completion_tokens", "Unknown")
                total_tokens = usage.get("total_tokens", "Unknown")
                
                print(f"Prompt tokens: {prompt_tokens}")
                print(f"Completion tokens: {completion_tokens}")
                print(f"Total tokens: {total_tokens}")
        else:
            print("\nNo valid response received.")
            print("Raw response:")
            print(json.dumps(response, indent=2))
    
    finally:
        # Clean up
        await service._close_session()


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="OpenRouter Service Example")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-32b-instruct", 
                        help="Model to use for generation")
    parser.add_argument("--prompt", type=str, 
                        default="Explain the concept of multi-step reasoning in AI models.",
                        help="Prompt to send to the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Run appropriate async function
    if args.list_models:
        asyncio.run(list_models())
    else:
        asyncio.run(chat_example(
            model=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        ))


if __name__ == "__main__":
    main() 