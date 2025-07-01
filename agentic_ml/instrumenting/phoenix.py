
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from .base import BaseInstrumentor

class PhoenixInstrumentor(BaseInstrumentor):
    """
    Instrumentor for Arize AI Phoenix.
    """

    def instrument(self):
        """
        Instruments the `smolagents` library with Phoenix.
        """
        register()
        SmolagentsInstrumentor().instrument()
        print("Successfully instrumented with Arize AI Phoenix.")
