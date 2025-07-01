
import os
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def setup_phoenix():
    """
    Sets up the Arize AI Phoenix instrumentor for smol-agents.
    """
    register()
    SmolagentsInstrumentor().instrument()
    print("Phoenix instrumentor setup complete.")

