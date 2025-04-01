# MSR Framework Usage Guide

This guide covers how to use the Multi-Step Reasoning (MSR) Framework for various research and reasoning tasks.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Command-Line Usage](#command-line-usage)
- [Python API Usage](#python-api-usage)
- [REST API Usage](#rest-api-usage)
- [Configuration Options](#configuration-options)
- [Models and Performance](#models-and-performance)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Clone the repository
git clone https://github.com/darwin-labs/MSR.git
cd MSR

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
cp .env.template .env
# Edit .env with your OpenRouter API key
```

## Getting Started

### Set up API Keys

1. Get an API key from [OpenRouter](https://openrouter.ai)
2. Add your API key to the `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
```

### Basic Example

Run a simple agent example:

```bash
python3 examples/agent_example.py "Research the latest advancements in quantum computing"
```

## Command-Line Usage

The `agent_example.py` script provides a simple way to interact with the MSR Framework from the command line.

### Command-Line Arguments

```bash
python3 examples/agent_example.py [TASK] [OPTIONS]
```

#### Required Arguments:
- `TASK`: The research task or question to investigate (quoted if it contains spaces)

#### Optional Arguments:

| Argument | Description |
|----------|-------------|
| `--enable-python` | Allow Python code execution |
| `--enable-terminal` | Allow terminal command execution |
| `--enable-web-search` | Allow web search functionality |
| `--enable-data-analysis` | Enable data analysis tools |
| `--steps NUM` | Maximum number of steps in the plan (default: 3) |
| `--temperature FLOAT` | Temperature for generation (default: 0.7) |
| `--model MODEL` | Model to use (default: google/gemini-2.0-flash-001) |
| `--output-file FILE` | Save agent state to a JSON file |
| `--log-level LEVEL` | Set logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `--log-file FILE` | Save logs to a file |
| `--help` | Show help message and exit |

### Usage Examples

Basic usage:
```bash
python3 examples/agent_example.py "Research the latest advancements in quantum computing"
```

Enable Python code execution:
```bash
python3 examples/agent_example.py "Analyze the performance of different sorting algorithms" --enable-python
```

Enable terminal commands:
```bash
python3 examples/agent_example.py "Analyze the disk usage in the current directory" --enable-terminal
```

Multiple tools with specified steps:
```bash
python3 examples/agent_example.py "Analyze the COVID-19 data trends in 2023" --enable-python --enable-data-analysis --steps 5
```

Save output to file:
```bash
python3 examples/agent_example.py "Research climate change impacts" --output-file results.json
```

Comprehensive example:
```bash
python3 examples/agent_example.py "Analyze Python code performance optimization techniques" --enable-python --enable-terminal --enable-web-search --enable-data-analysis --steps 5 --temperature 0.7 --model "google/gemini-2.0-flash-001" --output-file research_results.json --log-level INFO --log-file agent.log
```

## Python API Usage

You can use the MSR Framework directly in your Python code as a library.

### Basic Agent Usage

```python
import asyncio
from src.msr.agent import create_agent

async def run_agent():
    # Create an agent
    agent = create_agent(
        task="Research the impact of AI on healthcare",
        allowed_tools=["python", "web_search"],
        model="google/gemini-2.0-flash-001",
        temperature=0.7,
        save_state_path="agent_state.json"
    )
    
    # Generate a plan
    plan = await agent.generate_plan_async(num_steps=5)
    
    # Print plan details
    print(f"Plan: {plan.title}")
    print(f"Objective: {plan.objective}")
    print(f"Steps: {len(plan.steps)}")
    
    # Execute the plan
    results = await agent.execute_all_steps()
    
    # Print results
    for result in results:
        print(f"Step {result.step_id}: {'Success' if result.success else 'Failed'}")
        print(f"Findings: {result.findings[:100]}...")
        print(f"Learning: {result.learning}")
        print("-" * 50)

# Run the async function
asyncio.run(run_agent())
```

### Manual Step Execution

```python
import asyncio
from src.msr.agent import create_agent

async def run_agent_manual_steps():
    # Create an agent
    agent = create_agent(
        task="Create a data visualization of stock market trends",
        allowed_tools=["python", "data_analysis"],
        model="google/gemini-2.0-flash-001"
    )
    
    # Generate a plan
    await agent.generate_plan_async(num_steps=3)
    
    # Execute specific steps
    step1_result = await agent.execute_step_async(
        step_id="step1",
        allow_code_execution=True,
        allow_data_analysis=True
    )
    
    print(f"Step 1 results: {step1_result.findings[:100]}...")
    
    # Execute another step
    step2_result = await agent.execute_step_async(
        step_id="step2",
        allow_code_execution=True,
        allow_data_analysis=True
    )
    
    print(f"Step 2 results: {step2_result.findings[:100]}...")

# Run the async function
asyncio.run(run_agent_manual_steps())
```

## REST API Usage

To use the REST API, you need to start the server first.

### Starting the Server

```bash
python -m src.server.server
```

With custom configuration:
```bash
python -m src.server.server --host 127.0.0.1 --port 8000 --log-level DEBUG --log-file logs/debug.log
```

### API Client Examples

Using the Python client:

```python
from examples.server_client_example import MSRClient

# Create client instance
client = MSRClient("http://localhost:5000", api_key="your-api-key")

# Create an agent
agent_data = client.create_agent(
    task="Research the latest advancements in quantum computing",
    allowed_tools=["WEB_SEARCH", "PYTHON_EXECUTION"],
    model="google/gemini-2.0-flash-001",
    temperature=0.7
)

# Get agent ID
agent_id = agent_data["agent_id"]

# Generate a plan
plan = client.generate_plan(agent_id, max_steps=5)

# Execute the plan
results = client.execute_plan(agent_id)

# Print results
for step_result in results["results"]["execution_results"]:
    print(f"Step {step_result['step_id']}: {'Success' if step_result['success'] else 'Failed'}")
    print(f"Findings: {step_result['findings'][:100]}...")
```

Using curl:

```bash
# Create an agent
curl -X POST http://localhost:5000/api/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "task": "Research the latest advancements in quantum computing",
    "allowed_tools": ["WEB_SEARCH", "PYTHON_EXECUTION"],
    "model": "google/gemini-2.0-flash-001",
    "temperature": 0.7
  }'

# Generate a plan (replace AGENT_ID with the actual agent ID)
curl -X POST http://localhost:5000/api/agents/AGENT_ID/plan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "max_steps": 5
  }'

# Execute the plan
curl -X POST http://localhost:5000/api/agents/AGENT_ID/execute \
  -H "Authorization: Bearer your-api-key"
```

## Configuration Options

### Environment Variables

The MSR Framework uses these environment variables (can be set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | None |
| `DEFAULT_MODEL` | Default model to use | "google/gemini-2.0-flash-001" |
| `MSR_SERVER_HOST` | Server host address | "0.0.0.0" |
| `MSR_SERVER_PORT` | Server port | 5000 |
| `MSR_API_KEY` | API key for server authentication | None |
| `MSR_API_KEY_REQUIRED` | Whether API authentication is required | "true" |
| `MSR_LOG_LEVEL` | Logging level | "INFO" |
| `MSR_LOG_FILE` | Log file path | None |

### Runtime Configuration

You can override configuration at runtime:

```python
from src.utils.config import config_manager

# Set configuration values
config_manager.set("DEFAULT_MODEL", "google/gemini-2.0-flash-001")
config_manager.set_api_key("openrouter", "your-api-key-here")

# Get configuration values
model = config_manager.get("DEFAULT_MODEL")
api_key = config_manager.get_api_key("openrouter")
```

## Models and Performance

The MSR Framework supports various models through OpenRouter:

| Model | Strengths | Best For |
|-------|-----------|----------|
| anthropic/claude-3-opus | Strong reasoning, long context | Complex research, deep analysis |
| anthropic/claude-3-sonnet | Good balance of quality and efficiency | General tasks, mid-complexity |
| qwen/qwen2.5-32b-instruct | Good overall performance | Mixed research tasks |
| google/gemini-2.0-flash-001 | Fast, efficient, cost-effective | Simple tasks, quick answers |
| deepseek/deepseek-v3-base | Good performance, cost-effective | General reasoning, analysis |

### Choosing the Right Model

- For simple tasks, use `google/gemini-2.0-flash-001` for speed and efficiency.
- For balanced performance, use `anthropic/claude-3-sonnet`.
- For complex research, use `anthropic/claude-3-opus`.

## Error Handling

The MSR Framework implements comprehensive error handling with retries:

- Network errors are automatically retried with exponential backoff
- API rate limit errors are handled with appropriate delays
- JSON parsing errors are caught and formatted into useful error messages

### Common Error Messages

| Error | Description | Solution |
|-------|-------------|----------|
| "OpenRouter API error" | Error from the OpenRouter API | Check API key and rate limits |
| "Connection error" | Network connectivity issue | Check internet connection |
| "No plan generated" | Failed to generate a research plan | Refine task description |
| "JSON parsing error" | Failed to parse LLM output as JSON | Try different model or temperature |
| "Unclosed client session" | Session not properly closed | Use async context managers |

## Best Practices

### Writing Effective Prompts

- Be specific and clear about what you want to accomplish
- Break complex tasks into subtasks
- Provide relevant context and constraints
- Use domain-specific terminology when appropriate

Example:
```
"Research the impact of climate change on agricultural productivity in sub-Saharan Africa, focusing on changes in the last decade and potential mitigation strategies."
```

### Tool Selection

- Only enable tools that are necessary for the task
- For data analysis tasks, enable both Python code and data analysis tools
- For research tasks, enable web search
- For system tasks, enable terminal commands carefully

### Memory Management

- Save agent state for long-running tasks
- Use `save_state_path` parameter to enable automatic state saving
- Use the `output-file` option in command-line usage

## Advanced Usage

### Custom Tool Integration

You can integrate custom tools by extending the `Tool` class:

```python
from src.msr.agent import Tool, ToolType

# Create a custom tool
custom_tool = Tool(
    type=ToolType.PYTHON_CODE,
    name="custom_analysis",
    description="Perform custom data analysis",
    function=your_custom_function,
    arguments={"param1": "value1"},
    requires_approval=True
)

# Add tool to agent
agent = create_agent(
    task="Your research task",
    allowed_tools=["python", "web_search"]
)
agent.add_tool(custom_tool)
```

### Combining Multiple Agents

```python
import asyncio
from src.msr.agent import create_agent

async def multi_agent_research():
    # Create first agent for data collection
    data_agent = create_agent(
        task="Collect data on renewable energy adoption rates in Europe",
        allowed_tools=["web_search", "data_analysis"]
    )
    
    # Generate and execute plan
    await data_agent.run()
    
    # Use first agent's results for second agent
    findings = "\n".join([r.findings for r in data_agent.state.executed_steps if r.success])
    
    # Create second agent for analysis
    analysis_agent = create_agent(
        task="Analyze trends in renewable energy adoption and make projections",
        allowed_tools=["python", "data_analysis"]
    )
    
    # Add context from first agent
    await analysis_agent.run(additional_context=findings)
    
    # Print combined results
    print("=== Data Collection Results ===")
    print(data_agent.state.to_dict())
    
    print("\n=== Analysis Results ===")
    print(analysis_agent.state.to_dict())

# Run the async function
asyncio.run(multi_agent_research())
```

## Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues

**Issue**: "OpenRouter API error: Unauthorized"

**Solution**: 
- Check that your API key is valid
- Ensure the API key is correctly set in the `.env` file
- Try setting the API key directly in your code:
  ```python
  from src.utils.config import config_manager
  config_manager.set_api_key("openrouter", "your-api-key-here")
  ```

#### 2. Model Selection Issues

**Issue**: "Model not available" or poor quality results

**Solution**:
- Verify model name is correct with exact spelling
- Try switching to another model
- Check OpenRouter for available models: `print(service.get_available_models())`

#### 3. Execution Errors

**Issue**: "Unclosed client session" warnings

**Solution**:
- Use async context managers
- Ensure `close_session()` is called
- Use the `finally` block in try/except to close sessions

#### 4. JSON Parsing Errors

**Issue**: "JSON parsing error" during plan generation

**Solution**:
- Lower the temperature setting to get more deterministic outputs
- Try a different model
- Simplify the task description
- Increase max_tokens

#### 5. Performance Issues

**Issue**: Slow execution or timeouts

**Solution**:
- Reduce the number of steps in the plan
- Use a faster model
- Enable fewer tools
- Break complex tasks into subtasks

If you encounter persistent issues, check the logs for more detailed error information and refer to the documentation or open an issue on GitHub. 