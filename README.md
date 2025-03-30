# Multi-Step Reasoning (MSR) Framework

A framework to enhance the reasoning capabilities of foundation models through structured multi-step thinking processes.

## Overview

This project implements a Multi-Step Reasoning architecture designed to improve the problem-solving capabilities of base language models. By guiding models through explicit reasoning steps before generating final answers, we aim to reduce errors and enhance performance on complex tasks.

![Multi-Step Reasoning Architecture](static/assets/Multi-Step-Reasoning-Architecture-Horizontal.png)

## Core Model

The architecture is built around **Qwen QWQ 32B**, leveraging its capabilities while adding structured reasoning components.

## Key Features

- **Explicit reasoning steps** that break down complex problems
- **Chain-of-thought prompting** integrated into the model architecture
- **Self-verification mechanisms** to catch and correct errors
- **Planning components** that encourage the model to outline solution steps before execution

## Use Cases

- Complex mathematical reasoning
- Multi-hop question answering
- Logical puzzles and deduction tasks
- Programming and algorithmic problem solving

## Project Structure

- `src/`: Core implementation of the multi-step reasoning architecture
  - `models/`: Model configuration and adaptation for Qwen QWQ 32B
  - `reasoning/`: Implementation of reasoning steps and pipeline
  - `utils/`: Utility functions for prompts and other helpers
- `examples/`: Sample applications and demonstrations
- `eval/`: Evaluation frameworks and benchmarks
- `tests/`: Unit tests for components
- `scripts/`: Utility scripts for configuration and maintenance

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/darwin-labs/MSR.git
cd MSR

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Set up your API keys
python scripts/config_cli.py setup
```

### API Key Management

The MSR framework requires API keys to interact with language models like Qwen. We've implemented a secure mechanism for managing these keys:

1. Set up the initial configuration:

```bash
python scripts/config_cli.py setup
```

This will create a `.env` template file in the root directory. 

2. Edit the `.env` file to add your API keys:

```
QWEN_API_KEY="your-actual-key-here"
OPENAI_API_KEY="your-actual-key-here"
```

Alternatively, you can use the CLI tool to add keys securely:

```bash
python scripts/config_cli.py add-key qwen
# You'll be prompted to enter the key securely
```

3. View your configured API keys:

```bash
python scripts/config_cli.py list
```

4. Test if your API keys are working:

```bash
python scripts/config_cli.py test qwen
```

> **Note**: API keys are stored securely in `~/.msr/config.json` and will not be committed to version control. The `.env` file and configuration files are excluded in `.gitignore`.

### Running Examples

```bash
# Run the math problem solving example
python examples/math_problem_solving.py --problem "If x + y = 10 and x - y = 4, what is the value of x?"

# Run the evaluation benchmark
python eval/benchmark.py --output results.json
```

## Testing

```bash
# Run tests
python -m unittest discover tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 