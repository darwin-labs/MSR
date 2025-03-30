"""
MSR Flask Server Application

This module provides a REST API for the MSR framework using Flask.
"""

import os
import sys
import json
import uuid
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response
from werkzeug.middleware.proxy_fix import ProxyFix

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.msr.agent import Agent, ToolType, create_agent
from src.msr.planner_llm import PlannerLLM
from src.msr.logger import get_logger, configure_logger, LogLevel

# Store active agent sessions
active_agents = {}

def create_app(config=None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Configuration dictionary or path to config file
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Default configuration
    app.config.update(
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
    )
    
    # Apply additional configuration if provided
    if config:
        if isinstance(config, dict):
            app.config.update(config)
        elif isinstance(config, str) and os.path.exists(config):
            app.config.from_pyfile(config)
    
    # Configure logging for Flask
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    configure_logger(level=LogLevel[log_level.upper()])
    
    # For handling proxy headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    
    # Register error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "message": str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": str(error)}), 404
    
    @app.errorhandler(500)
    def server_error(error):
        return jsonify({"error": "Server error", "message": str(error)}), 500
    
    # Register API routes
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "version": "1.0.0",
            "active_agents": len(active_agents)
        })
    
    @app.route('/api/agents', methods=['POST'])
    def create_agent_endpoint():
        """Create a new agent session."""
        try:
            data = request.json
            
            # Validate request
            if not data:
                return jsonify({"error": "Missing request body"}), 400
            
            if 'task' not in data:
                return jsonify({"error": "Missing required field: task"}), 400
            
            # Parse allowed tools
            tool_types = []
            allowed_tools_raw = data.get('allowed_tools', [])
            
            if allowed_tools_raw:
                for tool_name in allowed_tools_raw:
                    try:
                        tool_types.append(ToolType[tool_name.upper()])
                    except KeyError:
                        return jsonify({
                            "error": f"Invalid tool type: {tool_name}",
                            "valid_tools": [t.name for t in ToolType]
                        }), 400
            
            # Generate agent ID
            agent_id = data.get('agent_id', str(uuid.uuid4()))
            
            # Create agent
            agent = create_agent(
                task=data['task'],
                allowed_tools=set(tool_types) if tool_types else None,
                agent_id=agent_id,
                model=data.get('model'),
                temperature=data.get('temperature', 0.7),
                require_step_approval=data.get('require_approval', True)
            )
            
            # Store agent in active sessions
            active_agents[agent_id] = {
                "agent": agent,
                "task": data['task'],
                "created_at": get_current_timestamp(),
                "status": "initialized",
                "plan": None,
                "executed_steps": []
            }
            
            return jsonify({
                "agent_id": agent_id,
                "task": data['task'],
                "status": "initialized",
                "allowed_tools": [t.name for t in tool_types] if tool_types else []
            }), 201
            
        except Exception as e:
            app.logger.exception("Error creating agent")
            return jsonify({"error": f"Error creating agent: {str(e)}"}), 500
    
    @app.route('/api/agents/<agent_id>', methods=['GET'])
    def get_agent_status(agent_id):
        """Get the status of an agent."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        agent_session = active_agents[agent_id]
        agent = agent_session["agent"]
        
        response = {
            "agent_id": agent_id,
            "task": agent_session["task"],
            "status": agent_session["status"],
            "created_at": agent_session["created_at"],
            "allowed_tools": [t.name for t in agent.allowed_tools] if agent.allowed_tools else []
        }
        
        # Add plan if available
        if agent.plan:
            response["plan"] = {
                "title": agent.plan.title,
                "objective": agent.plan.objective,
                "steps_count": len(agent.plan.steps)
            }
        
        # Add executed steps if available
        if agent.executed_steps:
            response["execution"] = {
                "steps_executed": len(agent.executed_steps),
                "steps_successful": len([s for s in agent.executed_steps if s.success]),
                "current_step": agent.current_step_index + 1 if agent.current_step_index is not None else None
            }
        
        return jsonify(response)
    
    @app.route('/api/agents/<agent_id>/plan', methods=['POST'])
    def generate_plan(agent_id):
        """Generate a plan for an agent."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        try:
            data = request.json or {}
            agent_session = active_agents[agent_id]
            agent = agent_session["agent"]
            
            # Extract planner config
            planner_config = {
                "model": data.get('model'),
                "temperature": data.get('temperature', 0.7),
                "max_steps": data.get('max_steps', 5),
                "domain": data.get('domain'),
                "additional_context": data.get('additional_context')
            }
            
            # Generate plan
            plan = agent.generate_plan(planner_config)
            
            # Update session status
            agent_session["status"] = "planned"
            agent_session["plan"] = plan
            
            # Convert plan to JSON-compatible format
            plan_dict = {
                "title": plan.title,
                "objective": plan.objective,
                "domain": plan.domain,
                "steps": [
                    {
                        "id": step.id,
                        "title": step.title,
                        "description": step.description,
                        "goal": step.goal,
                        "expected_output": step.expected_output,
                        "dependencies": step.dependencies
                    }
                    for step in plan.steps
                ]
            }
            
            return jsonify({
                "agent_id": agent_id,
                "status": "planned",
                "plan": plan_dict
            })
            
        except Exception as e:
            app.logger.exception("Error generating plan")
            return jsonify({"error": f"Error generating plan: {str(e)}"}), 500
    
    @app.route('/api/agents/<agent_id>/execute', methods=['POST'])
    async def execute_plan(agent_id):
        """Execute the agent's plan."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        agent_session = active_agents[agent_id]
        agent = agent_session["agent"]
        
        if not agent.plan:
            return jsonify({"error": "No plan has been generated for this agent"}), 400
        
        try:
            data = request.json or {}
            
            # Start execution
            agent_session["status"] = "executing"
            
            # Execute plan
            results = await agent.run()
            
            # Update session status
            agent_session["status"] = "completed"
            agent_session["executed_steps"] = agent.executed_steps
            
            # Format results
            execution_results = []
            for step_result in agent.executed_steps:
                result_dict = {
                    "step_id": step_result.step_id,
                    "success": step_result.success,
                    "findings": step_result.findings,
                    "learning": step_result.learning,
                }
                
                if step_result.error:
                    result_dict["error"] = step_result.error
                
                if step_result.next_steps:
                    result_dict["next_steps"] = step_result.next_steps
                
                if step_result.artifacts:
                    result_dict["artifacts"] = step_result.artifacts
                
                execution_results.append(result_dict)
            
            return jsonify({
                "agent_id": agent_id,
                "status": "completed",
                "results": {
                    "steps_total": len(agent.executed_steps),
                    "steps_successful": len([s for s in agent.executed_steps if s.success]),
                    "execution_results": execution_results
                }
            })
            
        except Exception as e:
            app.logger.exception("Error executing plan")
            agent_session["status"] = "error"
            return jsonify({"error": f"Error executing plan: {str(e)}"}), 500
    
    @app.route('/api/agents/<agent_id>/step/<step_id>/approve', methods=['POST'])
    def approve_step(agent_id, step_id):
        """Approve a step that's waiting for approval."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        agent_session = active_agents[agent_id]
        agent = agent_session["agent"]
        
        if not agent.pending_approval or agent.pending_approval.get('step_id') != step_id:
            return jsonify({"error": f"No pending approval for step {step_id}"}), 400
        
        try:
            # Approve the step
            agent.approve_step()
            
            return jsonify({
                "agent_id": agent_id,
                "step_id": step_id,
                "status": "approved"
            })
            
        except Exception as e:
            app.logger.exception("Error approving step")
            return jsonify({"error": f"Error approving step: {str(e)}"}), 500
    
    @app.route('/api/agents/<agent_id>/step/<step_id>/reject', methods=['POST'])
    def reject_step(agent_id, step_id):
        """Reject a step that's waiting for approval."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        agent_session = active_agents[agent_id]
        agent = agent_session["agent"]
        
        if not agent.pending_approval or agent.pending_approval.get('step_id') != step_id:
            return jsonify({"error": f"No pending approval for step {step_id}"}), 400
        
        try:
            data = request.json or {}
            reason = data.get('reason', 'Rejected by user')
            
            # Reject the step
            agent.reject_step(reason)
            
            return jsonify({
                "agent_id": agent_id,
                "step_id": step_id,
                "status": "rejected",
                "reason": reason
            })
            
        except Exception as e:
            app.logger.exception("Error rejecting step")
            return jsonify({"error": f"Error rejecting step: {str(e)}"}), 500
    
    @app.route('/api/agents/<agent_id>', methods=['DELETE'])
    def delete_agent(agent_id):
        """Delete an agent session."""
        if agent_id not in active_agents:
            return jsonify({"error": f"Agent not found: {agent_id}"}), 404
        
        try:
            # Remove agent from active sessions
            del active_agents[agent_id]
            
            return jsonify({
                "agent_id": agent_id,
                "status": "deleted"
            })
            
        except Exception as e:
            app.logger.exception("Error deleting agent")
            return jsonify({"error": f"Error deleting agent: {str(e)}"}), 500
    
    return app

def get_current_timestamp():
    """Get the current timestamp in ISO format."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + 'Z'

def run_server(host='0.0.0.0', port=5000, debug=False, config=None):
    """
    Run the Flask server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        debug: Whether to run in debug mode
        config: Configuration for the Flask app
    """
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server(debug=True) 