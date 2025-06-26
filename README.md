# Agentic_ML

## Overview
`Agentic_ML` is a sophisticated Python application designed to automate deep learning (DL) training pipelines using an agentic approach. It leverages the `smolagents` library from HuggingFace to orchestrate multiple specialized AI agents, each capable of handling different aspects of an ML project lifecycle. This includes tasks such as web browsing, PDF processing, file management, data inspection, and package installation.

The core of the system is the `AgentOrchestrator`, which manages the creation, configuration, and execution of these agents. It supports integration with various Large Language Models (LLMs) from providers like Hugging Face Transformers, Ollama, or OpenAI, allowing for flexible and powerful AI-driven automation.

## Features
- **Agentic Architecture**: Utilizes a hierarchical structure of AI agents to break down and execute complex ML tasks.
- **Configurable LLM Integration**: Supports multiple LLM providers (Transformers, Ollama, OpenAI) with customizable model IDs and provider-specific configurations.
- **Specialized Agents**: Includes agents for diverse functionalities such as:
    - Web Browsing
    - PDF Document Processing
    - File System Management
    - Data Inspection and Analysis
    - Package Installation
- **Automated Pipeline Execution**: Automates the end-to-end process of DL training pipelines, from data acquisition to model deployment.
- **Flexible Configuration**: Uses YAML configuration files to define LLM settings, agent behaviors, and project-specific parameters.
- **Detailed Logging**: Provides comprehensive logging to monitor agent activities and troubleshoot issues.

## Installation

### Prerequisites
- Python 3.8+

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Agentic_ML.git
   cd Agentic_ML
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, if you prefer using `pyproject.toml` with a tool like `pip install .` or `poetry install` (if using Poetry):
   ```bash
   pip install .
   ```

## Usage

To run the ML Pipeline Agent, you need to specify a configuration file and an initial directive.

First, ensure your `config.yaml` (or chosen config file) points to your initial directive file (e.g., `initial_directive_fmri.txt`). For example, in your `config.yaml`:

```yaml
run:
  initial_prompt: initial_directive_fmri.txt
  agent_working_dir: "../mri_voxel_model/"
```

Then, execute the main script:

```bash
python src/main.py --config configs/config.yaml
```

- `--config`: Path to the YAML configuration file for the agent pipeline (default: `configs/config.yaml`).

## License
[Specify your license here, e.g., MIT License]
