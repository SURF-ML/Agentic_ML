import os
import json
import traceback
import logging 
from typing import List, Dict, Any, Type, Optional

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from smolagents import CodeAgent, MultiStepAgent, LiteLLMModel
from smolagents.models import Model 
from smolagents import OpenAIServerModel


from agentic_ml.orchestrator.agent_definitions import AgentType, AGENT_SINGLE_TASK, ALL_TOOLS

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
        self.llm_provider = self.llm_config.get('provider', 'openai') 
        self.llm_provider_config = self.llm_config.get('provider_kwargs', {}) 

        self.model: Model = self._initialize_llm_model()
        self.agent: Optional[MultiStepAgent] = None
        self.initial_prompt_details: Optional[Dict[str, Any]] = None
    
        logger.info(f"AgentOrchestrator initialized with LLM model ID: {self.llm_model_id}")

        self.default_agent_config = {
            # if we don't import everything (*), it will bug out on a lot of libraries
            "additional_authorized_imports": ["*"],
            "stream_outputs": True,
            "max_steps": 50,
            "name": None,
            "description": None
        }

    def _initialize_llm_model(self) -> Model:
        """
        Initializes and returns the LLM model instance.
        """
        kwargs = self.llm_provider_config.get(self.llm_provider)
        try:
            logger.info(f"Attempting to initialize Transformer Model with model_id: {self.llm_model_id}")
            logger.info(f"Using LLM provider: {self.llm_provider}.")

            if self.llm_provider=="ollama":
                
                model_instance = LiteLLMModel(
                    model_id=self.llm_model_id, 
                    api_base=kwargs.get("api_base"),
                    num_ctx=kwargs.get("max_new_tokens"), # ollama default is 2048 which will fail horribly. 16384 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
                )

            elif self.llm_provider=="openai": # note that openai here stands for the OpenAIServerModel, we can use gemini with this model for instance
                model_instance = OpenAIServerModel(model_id=self.llm_model_id,
                                        api_key=os.environ[kwargs["api_key"]],
                                        api_base=kwargs.get("api_base")
                                    )
            
            logger.info(f"Transformer Model initialized successfully for model: {self.llm_model_id}")
            return model_instance
        except Exception as e:
            logger.error(f"Error initializing Transformer Model for model {self.llm_model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Transformer Model initialization failed for model {self.llm_model_id}: {e}")

    def _create_agent_instance(
        self,
        tools: List[callable],
        name: str,
        description: str,
        agent_class: Type[CodeAgent] = CodeAgent,
        managed_agents: Optional[List[CodeAgent]] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> CodeAgent:
        """Create any CodeAgent instance with proper config merging."""
        agent_config = self.default_agent_config.copy()
        agent_config["name"] = name
        agent_config["description"] = description

        if override_config:
            agent_config.update(override_config)
        
        try:
            instance = agent_class(
                tools=tools,
                model=self.model,
                managed_agents=managed_agents or [],
                **agent_config
            )
            logger.info(f"Agent '{name}' ({agent_class.__name__}) created successfully.")
            return instance
        except Exception as e:
            logger.error(f"Error creating agent '{name}': {e}", exc_info=True)
            raise

    def setup_main_orchestrator(self, orchestrator_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Sets up the root main orchestrator, can call multiple orchestrators
        """
        
        manager_description = (
                "You are the **Main Orchestrator Agent**, the master director of complex tasks.  "
                "Your primary function is not to perform tasks yourself, "
                "but to intelligently decompose a high-level goal into logical phases and delegate these phases to specialized agents."
                "You are the strategic brain of the operation."
            )
        
        # this class sets up orchestrator agents solely
        agent_type = AgentType.from_string("agent_orchestrating")
        orchestrator_tools = ALL_TOOLS # AGENT_SINGLE_TASK.get(agent_type)["tools"]

        # TODO: config hardcoded to None because we have a few fields that are incompatible with smolagents
        self.main_orchestrator = self._create_agent_instance(
            tools=orchestrator_tools,
            name="main_orchestrator",
            description=manager_description,
            agent_class=CodeAgent,
            #managed_agents=managed_orchestrators,
            override_config=None
        )
        logger.info(f"Manager agent '{self.main_orchestrator.name}' set up.")

    # For now only runs an orchestrator CodeAgent
    def run_orchestrator(
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
        if not self.main_orchestrator:
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
            final_output = self.main_orchestrator.run(directive, **run_kwargs)
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

