
from abc import ABC, abstractmethod

class BaseInstrumentor(ABC):
    """
    Abstract base class for instrumentors.
    """

    @abstractmethod
    def instrument(self):
        """
        Instruments the `smolagents` library.
        """
        pass
