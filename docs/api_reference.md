# MSR REST API Reference

This document provides a reference for the MSR REST API endpoints.

## Base URL

All API endpoints are relative to the base URL of your MSR server.

Example: `http://localhost:5000`

## Authentication

If API key authentication is enabled, you need to include the API key in your requests using one of these methods:

1. Bearer token in the Authorization header:
   ```
   Authorization: Bearer your-api-key
   ```

2. Query parameter:
   ```
   ?api_key=your-api-key
   ```

## Endpoints

### Health Check

```
GET /api/health
```

Check if the API server is running.

**Response**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_agents": 1
}
```

### Create Agent

```
POST /api/agents
```

Create a new agent session.

**Request Body**:
```json
{
  "task": "Analyze the performance trends in the S&P 500 over the past month",
  "allowed_tools": ["WEB_SEARCH", "DATA_ANALYSIS", "VISUALIZATION"],
  "model": "anthropic/claude-3-opus",
  "temperature": 0.7,
  "require_approval": true
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| task | string | The task for the agent to perform |
| allowed_tools | array | List of allowed tool types |
| model | string | Model to use for the agent |
| temperature | float | Temperature for generation |
| require_approval | boolean | Whether step execution requires approval |

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "task": "Analyze the performance trends in the S&P 500 over the past month",
  "status": "initialized",
  "allowed_tools": ["WEB_SEARCH", "DATA_ANALYSIS", "VISUALIZATION"]
}
```

### Get Agent Status

```
GET /api/agents/{agent_id}
```

Get the status of an agent.

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "task": "Analyze the performance trends in the S&P 500 over the past month",
  "status": "planned",
  "created_at": "2023-05-01T12:00:00Z",
  "allowed_tools": ["WEB_SEARCH", "DATA_ANALYSIS", "VISUALIZATION"],
  "plan": {
    "title": "S&P 500 Performance Analysis",
    "objective": "Analyze recent performance trends of the S&P 500",
    "steps_count": 3
  },
  "execution": {
    "steps_executed": 1,
    "steps_successful": 1,
    "current_step": 2
  }
}
```

### Generate Plan

```
POST /api/agents/{agent_id}/plan
```

Generate a plan for an agent.

**Request Body**:
```json
{
  "model": "anthropic/claude-3-opus",
  "temperature": 0.7,
  "max_steps": 5,
  "domain": "finance",
  "additional_context": "Focus on technology sector performance"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| model | string | Model to use for planning |
| temperature | float | Temperature for generation |
| max_steps | integer | Maximum number of steps in the plan |
| domain | string | Specific domain for the task |
| additional_context | string | Additional context for the task |

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "planned",
  "plan": {
    "title": "S&P 500 Performance Analysis",
    "objective": "Analyze recent performance trends of the S&P 500",
    "domain": "finance",
    "steps": [
      {
        "id": "step_1",
        "title": "Collect S&P 500 Data",
        "description": "Gather daily closing prices for the S&P 500 over the past month",
        "goal": "Obtain accurate historical data for analysis",
        "expected_output": "Dataset with date and closing price columns",
        "dependencies": []
      },
      {
        "id": "step_2",
        "title": "Calculate Key Metrics",
        "description": "Calculate daily returns, volatility, and moving averages",
        "goal": "Derive key performance metrics for analysis",
        "expected_output": "Enriched dataset with calculated metrics",
        "dependencies": ["step_1"]
      },
      {
        "id": "step_3",
        "title": "Visualize and Analyze Trends",
        "description": "Create visualizations and analyze the performance trends",
        "goal": "Identify and explain significant trends in the S&P 500",
        "expected_output": "Charts and analysis summary",
        "dependencies": ["step_2"]
      }
    ]
  }
}
```

### Execute Plan

```
POST /api/agents/{agent_id}/execute
```

Execute the plan for an agent.

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "results": {
    "steps_total": 3,
    "steps_successful": 3,
    "execution_results": [
      {
        "step_id": "step_1",
        "success": true,
        "findings": "Successfully collected S&P 500 data for the past month...",
        "learning": "The S&P 500 has shown significant volatility in the past 30 days"
      },
      {
        "step_id": "step_2",
        "success": true,
        "findings": "Calculated daily returns, 7-day and 14-day moving averages...",
        "learning": "The average daily return was 0.1% with a standard deviation of 0.7%"
      },
      {
        "step_id": "step_3",
        "success": true,
        "findings": "Created line charts showing price movement and key metrics...",
        "learning": "The index showed a generally positive trend with two significant dips"
      }
    ]
  }
}
```

### Approve Step

```
POST /api/agents/{agent_id}/step/{step_id}/approve
```

Approve a step that's waiting for approval.

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "step_id": "step_1",
  "status": "approved"
}
```

### Reject Step

```
POST /api/agents/{agent_id}/step/{step_id}/reject
```

Reject a step that's waiting for approval.

**Request Body**:
```json
{
  "reason": "This step uses incorrect data source"
}
```

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "step_id": "step_1",
  "status": "rejected",
  "reason": "This step uses incorrect data source"
}
```

### Delete Agent

```
DELETE /api/agents/{agent_id}
```

Delete an agent session.

**Response**:
```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "deleted"
}
```

## Error Responses

The API uses standard HTTP status codes to indicate the success or failure of a request:

- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Server Error

Error responses have the following format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

## Tool Types

The following tool types are available for agents:

- `PYTHON_EXECUTION`: Execute Python code
- `TERMINAL_COMMAND`: Execute terminal commands
- `WEB_SEARCH`: Search the web for information
- `FILE_OPERATIONS`: Read and write files
- `DATA_ANALYSIS`: Analyze data
- `VISUALIZATION`: Create visualizations 