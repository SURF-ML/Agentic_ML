import os
import logging
from typing import List, Dict

# Configure logger
logger = logging.getLogger(__name__)

class Directive:
    """
    Handles loading and formatting the main directive for the root orchestrator agent.
    This class separates the prompt content (stored in a file) from the operational code.
    """

    def __init__(self, template_path: str = "./directives/root_orchestrator_template.md"):
        """
        Initializes the Directive handler.

        Args:
            template_path (str): The file path to the markdown template for the directive.
        """
        self.template_path = template_path
        self.template_content = self._load_template()

    def _load_template(self) -> str:
        """
        Loads the prompt template from the specified file path.

        Returns:
            The content of the template file as a string.
        
        Raises:
            FileNotFoundError: If the template file cannot be found at the given path.
        """
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Directive template file not found at: {self.template_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the directive template: {e}")
            raise

    def get_directive(self, user_task: str) -> str:
        """
        Formats the loaded template with the specific user task.

        Args:
            user_task (str): The high-level goal or prompt from the user.

        Returns:
            The complete, formatted prompt ready to be used by the agent.
        """
        if "{initial_prompt}" not in self.template_content:
            logger.warning("The placeholder '{initial_prompt}' was not found in the template. The user task will not be injected.")
            return self.template_content

        # Using .format() to replace the placeholder with the actual user task
        return self.template_content.format(initial_prompt=user_task)