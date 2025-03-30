#!/usr/bin/env python3
"""
MSR Server Client Example

This script demonstrates how to interact with the MSR REST API using the requests library.
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, Any, Optional, List, Union
from pprint import pprint

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MSRClient:
    """Client for the MSR REST API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the MSR API server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set up authentication if API key is provided
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT requests
            params: Query parameters
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data, params=params)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            response.raise_for_status()
            
            # Return response data
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # Handle request errors
            print(f"Error making request to {url}: {str(e)}")
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Error response: {e.response.text}")
            
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the API server.
        
        Returns:
            Server health information
        """
        return self._make_request('GET', '/api/health')
    
    def create_agent(
        self, 
        task: str,
        allowed_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        require_approval: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new agent.
        
        Args:
            task: The task for the agent to perform
            allowed_tools: List of allowed tool types
            model: Model to use for the agent
            temperature: Temperature for generation
            require_approval: Whether step execution requires approval
            
        Returns:
            Agent information including agent ID
        """
        data = {
            'task': task,
            'allowed_tools': allowed_tools,
            'model': model,
            'temperature': temperature,
            'require_approval': require_approval
        }
        
        return self._make_request('POST', '/api/agents', data=data)
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the status of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent status information
        """
        return self._make_request('GET', f'/api/agents/{agent_id}')
    
    def generate_plan(
        self,
        agent_id: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_steps: int = 5,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a plan for an agent.
        
        Args:
            agent_id: ID of the agent
            model: Model to use for planning
            temperature: Temperature for generation
            max_steps: Maximum number of steps in the plan
            domain: Specific domain for the task
            additional_context: Additional context for the task
            
        Returns:
            Plan information
        """
        data = {
            'model': model,
            'temperature': temperature,
            'max_steps': max_steps,
            'domain': domain,
            'additional_context': additional_context
        }
        
        return self._make_request('POST', f'/api/agents/{agent_id}/plan', data=data)
    
    def execute_plan(self, agent_id: str) -> Dict[str, Any]:
        """
        Execute the plan for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Execution results
        """
        return self._make_request('POST', f'/api/agents/{agent_id}/execute')
    
    def approve_step(self, agent_id: str, step_id: str) -> Dict[str, Any]:
        """
        Approve a step that's waiting for approval.
        
        Args:
            agent_id: ID of the agent
            step_id: ID of the step
            
        Returns:
            Approval status
        """
        return self._make_request('POST', f'/api/agents/{agent_id}/step/{step_id}/approve')
    
    def reject_step(self, agent_id: str, step_id: str, reason: str = "Rejected by user") -> Dict[str, Any]:
        """
        Reject a step that's waiting for approval.
        
        Args:
            agent_id: ID of the agent
            step_id: ID of the step
            reason: Reason for rejection
            
        Returns:
            Rejection status
        """
        data = {'reason': reason}
        return self._make_request('POST', f'/api/agents/{agent_id}/step/{step_id}/reject', data=data)
    
    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Deletion status
        """
        return self._make_request('DELETE', f'/api/agents/{agent_id}')


def run_example(
    server_url: str,
    api_key: Optional[str] = None,
    task: str = "Analyze the performance trends in the S&P 500 over the past month",
    model: Optional[str] = None,
    tools: Optional[List[str]] = None,
    max_steps: int = 3,
    require_approval: bool = True,
    wait_for_completion: bool = True
):
    """
    Run an example workflow with the MSR API.
    
    Args:
        server_url: URL of the MSR API server
        api_key: API key for authentication
        task: Task for the agent to perform
        model: Model to use for the agent
        tools: List of tools to enable
        max_steps: Maximum number of steps in the plan
        require_approval: Whether to require approval for step execution
        wait_for_completion: Whether to wait for the agent to complete execution
    """
    # Default tools if none specified
    if tools is None:
        tools = ['WEB_SEARCH', 'DATA_ANALYSIS', 'VISUALIZATION']
    
    # Create client
    client = MSRClient(server_url, api_key)
    
    try:
        # Check server health
        print("Checking server health...")
        health = client.health_check()
        print(f"Server status: {health['status']}")
        print(f"Active agents: {health['active_agents']}")
        print()
        
        # Create agent
        print(f"Creating agent for task: {task}")
        agent_response = client.create_agent(
            task=task,
            allowed_tools=tools,
            model=model,
            temperature=0.7,
            require_approval=require_approval
        )
        
        agent_id = agent_response['agent_id']
        print(f"Agent created with ID: {agent_id}")
        print(f"Status: {agent_response['status']}")
        print(f"Allowed tools: {', '.join(agent_response['allowed_tools'])}")
        print()
        
        # Generate plan
        print("Generating plan...")
        plan_response = client.generate_plan(
            agent_id=agent_id,
            model=model,
            temperature=0.7,
            max_steps=max_steps
        )
        
        print(f"Plan generated: {plan_response['plan']['title']}")
        print(f"Objective: {plan_response['plan']['objective']}")
        print("\nSteps:")
        
        for step in plan_response['plan']['steps']:
            print(f"  {step['id']}: {step['title']}")
            print(f"    Goal: {step['goal']}")
            print(f"    Dependencies: {', '.join(step['dependencies']) if step['dependencies'] else 'None'}")
        
        print()
        
        # Execute plan
        print("Executing plan...")
        
        try:
            execution_response = client.execute_plan(agent_id=agent_id)
            
            print("Execution completed!")
            print(f"Steps completed: {execution_response['results']['steps_successful']}/{execution_response['results']['steps_total']}")
            
            # Print results
            print("\nExecution results:")
            for result in execution_response['results']['execution_results']:
                print(f"  Step {result['step_id']}: {'✓' if result['success'] else '✗'}")
                print(f"    Findings: {result['findings'][:100]}..." if len(result['findings']) > 100 else f"    Findings: {result['findings']}")
                print(f"    Learning: {result['learning']}")
                if not result['success'] and 'error' in result:
                    print(f"    Error: {result['error']}")
                print()
        
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            
            # Check agent status
            status = client.get_agent_status(agent_id)
            print(f"Agent status: {status['status']}")
            
            if 'execution' in status:
                print(f"Steps executed: {status['execution']['steps_executed']}")
                print(f"Steps successful: {status['execution']['steps_successful']}")
                print(f"Current step: {status['execution']['current_step']}")
        
        # Delete agent when done
        print("Deleting agent...")
        delete_response = client.delete_agent(agent_id)
        print(f"Agent deleted: {delete_response['status']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Parse command line arguments and run the example."""
    parser = argparse.ArgumentParser(description="Example client for the MSR API")
    
    # Server configuration
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='URL of the MSR API server (default: http://localhost:5000)')
    parser.add_argument('--api-key', type=str,
                       help='API key for authentication')
    
    # Task configuration
    parser.add_argument('--task', type=str, 
                       default="Analyze the performance trends in the S&P 500 over the past month",
                       help='Task for the agent to perform')
    parser.add_argument('--model', type=str,
                       help='Model to use for the agent')
    parser.add_argument('--tools', type=str, nargs='+',
                       default=['WEB_SEARCH', 'DATA_ANALYSIS', 'VISUALIZATION'],
                       help='Tools to enable for the agent')
    parser.add_argument('--max-steps', type=int, default=3,
                       help='Maximum number of steps in the plan')
    parser.add_argument('--no-approval', action='store_true',
                       help='Disable approval requirement for step execution')
    
    args = parser.parse_args()
    
    # Run the example
    run_example(
        server_url=args.server,
        api_key=args.api_key,
        task=args.task,
        model=args.model,
        tools=args.tools,
        max_steps=args.max_steps,
        require_approval=not args.no_approval
    )


if __name__ == '__main__':
    main() 