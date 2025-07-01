
# Instrumenting smol-agents with Arize AI Phoenix

This directory contains scripts to instrument `smol-agents` using Arize AI Phoenix for local inspection of agent runs.

## Option 1: Arize AI Phoenix (Local Inspection)

Phoenix is a lightweight, open-source tool for collecting and visualizing telemetry data on your local machine.

### Prerequisites

1.  **Install Dependencies**:
    Make sure you have the necessary packages installed.
    ```bash
    pip install 'smolagents[telemetry,toolkit]'
    ```

### How to Run

1.  **Start the Phoenix Server**:
    Run the following command in your terminal to start the Phoenix server in the background:
    ```bash
    python -m phoenix.server.main serve
    ```

2.  **Instrument Your Agent Code**:
    In your Python script where you run your `smol-agents` application, add the following code to set up the instrumentor:

    ```python
    from agentic_ml.instrumenting.phoenix_instrumentor import setup_phoenix

    # Set up Phoenix instrumentor
    setup_phoenix()

    # Your smol-agents application code here
    # ...
    ```

3.  **Run Your Agent**:
    Execute your Python script as you normally would.

4.  **Visualize Traces**:
    Open your web browser and navigate to `http://127.0.0.1:6006` to view the traces in the Phoenix UI.

