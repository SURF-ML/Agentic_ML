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

from agentic_ml.orchestrator.agent_definitions import AgentType, AGENT_SINGLE_TASK, AGENT_ORCHESTRATOR

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
        self.llm_provider_config = self.llm_config.get('provider_kwargs', {}) 

        self.model: Model = self._initialize_llm_model()
        self.agent: Optional[MultiStepAgent] = None
        self.initial_prompt_details: Optional[Dict[str, Any]] = None
    
        logger.info(f"AgentOrchestrator initialized with LLM model ID: {self.llm_model_id}")

        self.default_agent_config = {
            # "additional_authorized_imports": ["os", 
            #                                   "json", 
            #                                   "sys", 
            #                                   "collections", 
            #                                   "glob", 
            #                                   "shutil",
            #                                   "pandas",
            #                                   "pandas.*",
            #                                   "numpy", 
            #                                   "numpy.*", 
            #                                   "PIL", 
            #                                   "PIL.*", 
            #                                   "matplotlib", 
            #                                   "matplotlib.*", 
            #                                   "scipy",
            #                                   "scipy.*",
            #                                   "h5py",
            #                                   "sklearn",
            #                                   "sklearn.*",
            #                                   "torch", 
            #                                   "torch.*",
            #                                   "torchvision", 
            #                                   "tensorflow", 
            #                                   "logging",
            #                                   "posixpath",

            # ],
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
            if self.llm_provider=="transformers":
                model_instance = TransformersModel(model_id=self.llm_model_id, 
                                                   **kwargs)

            elif self.llm_provider=="ollama":
                
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

    def load_from_txt(self, filepath: str) -> bool:
        """
        Loads initial prompt details from a specified text file and stores them as a single string.

        Args:
            filepath (str): The path to the text file containing initial prompt details.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.initial_prompt_details = f.read()
            logger.info(f"Successfully loaded initial prompt details from: {filepath}")
            return True
        except FileNotFoundError:
            logger.error(f"Initial prompt text file not found at: {filepath}")
            self.initial_prompt_details = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading prompt from {filepath}: {e}", exc_info=True)
            self.initial_prompt_details = None
            return False

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
	
	
    # TODO: merge this function with _create_agent_instance()
    # We don't really use this outside of this orchestrator class anyway
    def create_specialized_agent(
        self,
        agent_type_str: str,
        agent_class: Type[CodeAgent] = CodeAgent,
        override_config: Optional[Dict[str, Any]] = None
    ) -> CodeAgent:
        """
        Creates a single specialized agent based on its type string.
        """
        try:
            agent_type = AgentType.from_string(agent_type_str)
        except ValueError as e:
            logger.error(str(e))
            raise
        
        spec = AGENT_SINGLE_TASK.get(agent_type)
        if not spec:
            raise ValueError(f"No specification found for agent type: {agent_type}")
        return self._create_agent_instance(
            tools=spec["tools"],
            name=spec["name"],
            description=spec["description"],
            agent_class=agent_class,
            override_config=override_config
        )

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

    def setup_orchestrator(
        self,
        manager_agent_class: Type[CodeAgent] = CodeAgent,
        manager_name: str = "agent_orchestrating",
        manager_description: Optional[str] = None,
        manager_override_config: Optional[Dict[str, Any]] = None,
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> CodeAgent:
        """
        Sets up a manager agent with a list of specialized sub-agents.
        The created manager agent is assigned to `self.agent`.

        Args:
            specialized_agent_type_strings: List of strings like ["Browse", "pdf_opening"].
            manager_agent_class: The class for the manager agent.
            manager_name: Name for the manager agent.
            manager_description: Description for the manager. If None, a default is used.
            manager_override_config: Additional config for the manager.
            specialized_agent_override_configs: Dict mapping agent_type_str to its override_config.
        """
        managed_agents: List[CodeAgent] = []
        agent_type = AgentType.from_string(manager_name)
        specialized_agent_strings = AGENT_ORCHESTRATOR.get(agent_type)["managed_agents"]

        if agent_configs is None:
            agent_configs = {}

        for agent_type_str in specialized_agent_strings:
            try:
                override_cfg = agent_configs.get(agent_type_str)
                # TODO: config hardcoded to None because we have a few fields that are incompatible with smolagents
                agent = self.create_specialized_agent(agent_class=manager_agent_class,
                                                      agent_type_str=agent_type_str, 
                                                      override_config=None)
                managed_agents.append(agent)
            except Exception as e:
                logger.error(f"Failed to create specialized agent of type '{agent_type_str}': {e}", exc_info=True)
                # Decide behavior: raise, or log and continue
                raise ValueError(f"Setup failed for specialized agent: {agent_type_str}") from e

        if manager_description is None:
            manager_description = (
                "I am a coordinator agent. I can delegate tasks to specialized agents "
                "for web Browse, PDF processing, file searching/managing, data inspection, and package installing. "
                "Clearly state your high-level goal, and I will manage the sub-tasks."
            )

        # this class sets up orchestrator agents solely
        # get default agent orchestrating tools
        orchestrator_tools = AGENT_SINGLE_TASK.get(AgentType.from_string("agent_orchestrating"))["tools"]

        agent = self._create_agent_instance(
            tools=orchestrator_tools,
            name=manager_name,
            description=manager_description,
            agent_class=manager_agent_class,
            managed_agents=managed_agents,
            override_config=manager_override_config
        )
        logger.info(f"Manager agent '{agent.name}' set up with {len(managed_agents)} specialized agents.")

        return agent


    def setup_main_orchestrator(self,
                                orchestrator_agent_strings: List[str],
                                orchestrator_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Sets up the root main orchestrator, can call multiple orchestrators
        """
        
        managed_orchestrators: List[CodeAgent] = []
        if orchestrator_configs is None:
            orchestrator_configs = {}
        for agent_type_str in orchestrator_agent_strings:
            try:
                override_cfg = orchestrator_configs.get(agent_type_str)
                agent_type = AgentType.from_string(agent_type_str)
                manager_description = AGENT_ORCHESTRATOR.get(agent_type)["description"]
                # TODO: config hardcoded to None because we have a few fields that are incompatible with smolagents
                orchestrator_agent = self.setup_orchestrator(manager_agent_class=CodeAgent,
                                                             manager_name=agent_type_str, 
                                                             manager_description=manager_description,
                                                             agent_configs=None)
                
                managed_orchestrators.append(orchestrator_agent)
            except Exception as e:
                logger.error(f"Failed to create specialized agent of type '{agent_type_str}': {e}", exc_info=True)
                # Decide behavior: raise, or log and continue
                raise ValueError(f"Setup failed for specialized agent: {agent_type_str}") from e
        
        manager_description = (
                "You are the **Main Orchestrator Agent**, the master director of a complex, end-to-end machine learning project.  "
                "Your primary function is not to perform tasks yourself, but to intelligently decompose a high-level goal into logical phases and delegate these phases to specialized **Role Orchestrator Agents**."
                "You are the strategic brain of the operation."
            )
        
        # this class sets up orchestrator agents solely
        agent_type = AgentType.from_string("agent_orchestrating")
        orchestrator_tools = AGENT_SINGLE_TASK.get(agent_type)["tools"]

        # TODO: config hardcoded to None because we have a few fields that are incompatible with smolagents
        self.main_orchestrator = self._create_agent_instance(
            tools=orchestrator_tools,
            name="main_orchestrator",
            description=manager_description,
            agent_class=CodeAgent,
            managed_agents=managed_orchestrators,
            override_config=None
        )
        logger.info(f"Manager agent '{self.main_orchestrator.name}' set up with {len(managed_orchestrators)} specialized agents.")

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

    # DEPRECATED
    def setup_singular_agent(
        self,
        tools: List[callable],
        name: str,
        description: str,
        agent_class: Type[CodeAgent] = CodeAgent,
        override_config: Optional[Dict[str, Any]] = None,
        set_as_main_agent: bool = False
    ) -> CodeAgent:
        """
        Initializes and sets up a single agent instance.
        If set_as_main_agent is True, it overwrites self.agent.
        """
        agent = self._create_agent_instance(
            tools=tools,
            name=name,
            description=description,
            agent_class=agent_class,
            override_config=override_config
        )
        if set_as_main_agent:
            self.agent = agent
            logger.info(f"Singular agent '{agent.name}' set as the main agent.")
        return agent
