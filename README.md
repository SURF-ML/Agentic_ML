# Agentic_ML

## Overview
`Agentic_ML` leverages the `smolagents` library from HuggingFace to orchestrate multiple specialized AI agents, spawned by a single agent orchestrator. This could include tasks such as web browsing, PDF processing, file management, data inspection, and package installation.

The core of the system is the `AgentOrchestrator`, which manages the creation, configuration, and execution of these agents. It supports integration with various Large Language Models (LLMs) from providers like Ollama, or OpenAI.

## Installation

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

## Quick Run

To quickly run the agent and see it in action, use the following command:

```bash
python -m agentic_ml.main --config ./agentic_ml/configs/config.yaml --prompt 'Hi could you open the following webpage and extract details from it: https://arxiv.org/pdf/2407.00203, make a detailed and thorough structured report in mark down.'
```

## License
[Specify your license here, e.g., MIT License]


