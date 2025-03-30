"""
MSR Server Middleware

Middleware components for the MSR REST API server, including:
- API Key Authentication
- CORS handling
- Request logging
"""

import os
import time
import functools
from typing import Optional, Callable, Dict, Any
from flask import Flask, request, jsonify, Response, g
from flask_cors import CORS

from src.server.config import (
    ENABLE_CORS, CORS_ORIGINS, API_KEY_REQUIRED, API_KEY, LOG_LEVEL
)
from src.msr.logger import get_logger, LogEventType


# Get logger instance
logger = get_logger()


def setup_middleware(app: Flask) -> None:
    """
    Set up middleware for the Flask application.
    
    Args:
        app: Flask application instance
    """
    # Set up CORS if enabled
    if ENABLE_CORS:
        CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
    
    # Set up request logging
    @app.before_request
    def log_request_info():
        """Log details of each incoming request."""
        g.start_time = time.time()
        
        # Skip logging for health check endpoint to avoid spam
        if request.path == '/api/health':
            return
        
        logger.debug(
            message=f"Request received: {request.method} {request.path}",
            event_type=LogEventType.SERVER_REQUEST,
            context={
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr,
                "user_agent": request.user_agent.string if request.user_agent else None,
                "content_length": request.content_length
            }
        )
    
    @app.after_request
    def log_response_info(response: Response) -> Response:
        """Log details of each outgoing response."""
        # Skip logging for health check endpoint to avoid spam
        if request.path == '/api/health':
            return response
        
        duration = time.time() - g.get('start_time', time.time())
        
        logger.debug(
            message=f"Response sent: {response.status_code}",
            event_type=LogEventType.SERVER_RESPONSE,
            context={
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "content_length": response.content_length,
                "duration_ms": round(duration * 1000, 2)
            }
        )
        return response
    
    # Set up API key authentication if required
    if API_KEY_REQUIRED and API_KEY:
        @app.before_request
        def authenticate():
            """Check API key for all /api/ endpoints except health check."""
            # Skip API key check for health check endpoint
            if request.path == '/api/health':
                return
            
            # Check if the path starts with /api/
            if request.path.startswith('/api/'):
                # Get API key from Authorization header or query parameter
                api_key = None
                auth_header = request.headers.get('Authorization')
                
                if auth_header and auth_header.startswith('Bearer '):
                    api_key = auth_header.split(' ')[1]
                else:
                    api_key = request.args.get('api_key')
                
                # Check if API key matches
                if not api_key or api_key != API_KEY:
                    logger.warning(
                        message="Invalid API key",
                        event_type=LogEventType.SECURITY_VIOLATION,
                        context={
                            "method": request.method,
                            "path": request.path,
                            "remote_addr": request.remote_addr,
                            "user_agent": request.user_agent.string if request.user_agent else None
                        }
                    )
                    
                    return jsonify({
                        "error": "Unauthorized",
                        "message": "Invalid API key"
                    }), 401


def require_api_key(func: Callable) -> Callable:
    """
    Decorator to require API key for specific endpoints.
    Use this for endpoints that need API key authentication when global authentication is disabled.
    
    Args:
        func: The route function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if not API_KEY_REQUIRED or not API_KEY:
            return func(*args, **kwargs)
        
        # Get API key from Authorization header or query parameter
        api_key = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        else:
            api_key = request.args.get('api_key')
        
        # Check if API key matches
        if not api_key or api_key != API_KEY:
            logger.warning(
                message="Invalid API key",
                event_type=LogEventType.SECURITY_VIOLATION,
                context={
                    "method": request.method,
                    "path": request.path,
                    "remote_addr": request.remote_addr,
                    "user_agent": request.user_agent.string if request.user_agent else None
                }
            )
            
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid API key"
            }), 401
        
        return func(*args, **kwargs)
    
    return decorated_function


def cleanup_session(app: Flask, cleanup_interval: int = 300) -> None:
    """
    Set up a scheduled task to clean up expired agent sessions.
    
    Args:
        app: Flask application instance
        cleanup_interval: Interval in seconds for cleanup (default: 300)
    """
    from threading import Thread
    import time
    from datetime import datetime, timedelta
    
    from src.server.app import active_agents
    from src.server.config import SESSION_TIMEOUT
    
    def cleanup_task():
        """Periodically clean up expired agent sessions."""
        while True:
            try:
                with app.app_context():
                    # Calculate expiration time
                    expiration_time = datetime.utcnow() - timedelta(seconds=SESSION_TIMEOUT)
                    expired_agents = []
                    
                    # Find expired agents
                    for agent_id, session in active_agents.items():
                        if 'created_at' in session:
                            created_time = datetime.fromisoformat(session['created_at'].replace('Z', '+00:00'))
                            if created_time < expiration_time:
                                expired_agents.append(agent_id)
                    
                    # Remove expired agents
                    for agent_id in expired_agents:
                        logger.info(
                            message=f"Cleaning up expired agent session: {agent_id}",
                            event_type=LogEventType.SERVER_MAINTENANCE,
                            context={"agent_id": agent_id}
                        )
                        del active_agents[agent_id]
                    
                    # Log stats
                    if expired_agents:
                        logger.info(
                            message=f"Cleaned up {len(expired_agents)} expired agent sessions",
                            event_type=LogEventType.SERVER_MAINTENANCE,
                            context={"active_sessions": len(active_agents)}
                        )
            
            except Exception as e:
                logger.error(
                    message=f"Error in session cleanup task: {str(e)}",
                    event_type=LogEventType.ERROR
                )
            
            # Sleep until next cleanup
            time.sleep(cleanup_interval)
    
    # Start cleanup thread
    cleanup_thread = Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start() 