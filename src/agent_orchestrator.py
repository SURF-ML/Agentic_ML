import os
import json
import traceback
import logging 
from typing import List, Dict, Any, Type, Optional

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from smolagents import CodeAgent, MultiStepAgent, LiteLLMModel
from smolagents.models import TransformersModel, Model 
from smolagents import OpenAIServerModel


class AgentOrchestrator:
    """
    Orchestrates the lifecycle of an agent, including its initialization,
    loading initial prompt data, running specific phases with directives,
    and managing its LLM model.
    """

    def __init__(self, config: Dict[str, any]):
        """
        Initializes the AgentOrchestrator with LLM configuration.

        Args:
            llm_model_id (str): The ID of the LLM model to be used (e.g., "gpt-4o").
            llm_provider_config (Dict[str, Any], optional): Additional configuration for the LLM provider.
        """
        # 1. Initialize LLM Model from Config
        self.llm_config = config.get('llm', {})
        self.llm_model_id = self.llm_config.get('model_id', 'Qwen/Qwen3-4B') 
        self.llm_provider = self.llm_config.get('provider', 'transformers') 
        self.llm_provider_config = self.llm_config.get('transformers_model_kwargs', {}) 

        self.model: Model = self._initialize_llm_model()
        self.agent: Optional[MultiStepAgent] = None
        self.initial_prompt_details: Optional[Dict[str, Any]] = None
    
        logger.info(f"AgentOrchestrator initialized with LLM model ID: {self.llm_model_id}")

    def _initialize_llm_model(self) -> Model:
        """
        Initializes and returns the LLM model instance.
        """
        try:
            logger.info(f"Attempting to initialize Transformer Model with model_id: {self.llm_model_id}")
            logger.info(f"Using LLM provider: {self.llm_provider}.")
            if self.llm_provider=="transformers":
                    
                model_instance = TransformersModel(model_id=self.llm_model_id, **self.llm_provider_config)

            elif self.llm_provider=="ollama":
                    
                model_instance = LiteLLMModel(
                    model_id=self.llm_model_id, 
                    api_base="http://localhost:11434",
                    api_key="YOUR_API_KEY", 
                    num_ctx=self.llm_provider_config.get("kwargs").get("max_new_tokens"), # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
                )

            elif self.llm_provider=="openai": # note that openai here stands for the OpenAIServerModel, we can use gemini with this model for instance
                model_instance = OpenAIServerModel(model_id="gemini-2.0-flash",
                                        api_key=os.environ['GEMINI_API_KEY'],
                                        # Google Gemini OpenAI-compatible API base URL
                                        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                                    )
            
            logger.info(f"Transformer Model initialized successfully for model: {self.llm_model_id}")
            return model_instance
        except Exception as e:
            logger.error(f"Error initializing Transformer Model for model {self.llm_model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Transformer Model initialization failed for model {self.llm_model_id}: {e}")

    def load_from_json(self, filepath: str) -> bool:
        """
        Loads initial prompt details from a specified JSON file and stores them.

        Args:
            filepath (str): The path to the JSON file containing initial prompt details.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.initial_prompt_details = json.load(f)
            logger.info(f"Successfully loaded initial prompt details from: {filepath}")
            return True
        except FileNotFoundError:
            logger.error(f"Initial prompt JSON file not found at: {filepath}")
            self.initial_prompt_details = None
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {filepath}: {e}")
            self.initial_prompt_details = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading prompt from {filepath}: {e}", exc_info=True)
            self.initial_prompt_details = None
            return False


    def get_initial_prompt_details(self) -> Optional[Dict[str, Any]]:
        """
        Returns the loaded initial prompt details.

        Returns:
            Optional[Dict[str, Any]]: The dictionary of prompt details, or None if not loaded.
        """
        if self.initial_prompt_details is None:
            logger.warning("Attempted to get initial prompt details, but none have been loaded.")
        return self.initial_prompt_details

    def setup_agent(
        self,
        list_of_tools: List[callable],
        agent_class: Type[MultiStepAgent] = CodeAgent,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes and sets up an agent instance.

        Args:
            list_of_tools (List[callable]): A list of tool functions callable by the agent.
            agent_class (Type[MultiStepAgent]): The class of the agent to instantiate. Defaults to CodeAgent.
            agent_config (Dict[str, Any], optional): Configuration dictionary for the agent.
        """
        effective_agent_config = agent_config or {}
        agent_name = agent_class.__name__

        default_configs = {
            "additional_authorized_imports": [
                "os", "json", "sys", "collections", "glob", "shutil",
                "pandas", "numpy", "PIL", "matplotlib", "sklearn",
                "torch", "torchvision", "tensorflow", "logging" 
            ],
            "stream_outputs": True,
            "max_steps": 50 
        }
        # Merge, with agent_config taking precedence
        merged_config = {**default_configs, **effective_agent_config}

        try:
            self.agent = agent_class(
                tools=list_of_tools,
                model=self.model,
                **merged_config
            )
            model_id_str = self.model.model_id if hasattr(self.model, 'model_id') else 'N/A'
            logger.info(f"{agent_name} initialized successfully with {len(list_of_tools)} tools, model {model_id_str}, and config: {merged_config}")
        except Exception as e:
            logger.error(f"Error initializing {agent_name} with config {merged_config}: {e}", exc_info=True)
            raise RuntimeError(f"{agent_name} initialization failed: {e}")

    def run_agent(
        self,
        directive: str,
        task_images: Optional[List[Any]] = None,
        additional_run_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Runs the initialized agent with a given directive for a specific phase.

        Args:
            directive (str): The task prompt/directive for the agent for this phase.
            task_images (List[Any], optional): List of PIL Image objects if the task involves images.
            additional_run_args (Dict[str, Any], optional): Additional arguments for agent's run method.

        Returns:
            Any: The final answer or output from the agent's run for this phase.
        """
        if not self.agent:
            logger.error("Agent has not been set up. Call setup_agent() first.")
            raise RuntimeError("Agent has not been set up. Call setup_agent() first.")

        agent_name = self.agent.__class__.__name__
        logger.info(f"Starting agent phase for {agent_name}.")
        # Avoid logging the full directive if it's very long; could be configured
        logger.debug(f"Directive for {agent_name} (first 300 chars): {directive[:300]}...")

        run_kwargs = {}
        if task_images:
            run_kwargs["images"] = task_images
        if additional_run_args:
            run_kwargs["additional_args"] = additional_run_args
        
        logger.debug(f"Calling {agent_name}.run() with kwargs: {run_kwargs}")

        try:
            final_output = self.agent.run(directive, **run_kwargs)
            logger.info(f"Agent phase for {agent_name} finished.")
            if final_output is not None:
                final_output_str = str(final_output)
                logger.debug(f"Final output from {agent_name} (first 500 chars):\n{final_output_str[:500]}{'...' if len(final_output_str) > 500 else ''}")
            else:
                logger.info(f"{agent_name} run finished with no explicit final output.")
            return final_output
        except Exception as e:
            logger.error(f"An error occurred during {agent_name} execution phase: {e}", exc_info=True)
            # Potentially return a structured error object instead of a string
            return {"error": True, "message": f"Agent execution phase failed: {str(e)}", "details": traceback.format_exc()}

    def get_agent_instance(self) -> Optional[MultiStepAgent]:
        """Returns the current agent instance."""
        return self.agent

    def get_llm_model_instance(self) -> Model:
        """Returns the LLM model instance."""
        return self.model
