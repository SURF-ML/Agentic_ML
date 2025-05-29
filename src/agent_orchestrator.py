import os
import json
import traceback
import logging 
from enum import Enum
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

from agent_environment.agent.tools import (
    create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, search_arxiv, search_github_repositories,
    read_scratchpad, update_scratchpad, inspect_file_type_and_structure,
    search_wikipedia, ask_user_for_input, check_python_package_version, 
    list_installed_python_packages, grep_directory, zip_files, unzip_file,
    search_google_scholar, download_file_from_url, read_pdf_content,
    find_files_by_pattern

)

class AgentType(Enum):
    Browsing = "browsing"
    PDF_OPENING = "pdf_opening"
    FILE_SEARCHING = "file_searching"
    DATA_INSPECTING = "data_inspecting"
    PACKAGE_INSTALLING = "package_installing"
    FILE_MANAGING = "file_managing"
    AGENT_ORCHESTRATING = "agent_orchestrating"
    
    @classmethod
    def from_string(cls, s: str) -> 'AgentType':
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown agent type string: {s}")

AGENT_SPECIFICATIONS = {
    AgentType.Browsing: {
        "tools": [
            browse_webpage,
            search_wikipedia,
            search_arxiv,
            search_github_repositories,
            search_google_scholar,
            download_file_from_url,
            create_directory
        ],
        "name": "web_navigator_agent",
        "description": "Specialized in Browse websites, using online search engines (Wikipedia, arXiv, GitHub, Google Scholar), and downloading files from URLs. Args: query (for searches) or url (for Browse/downloading)."
    },
    AgentType.PDF_OPENING: {
        "tools": [
            read_pdf_content,
            find_files_by_pattern, 
            inspect_file_type_and_structure, 
            read_file_content
        ],
        "name": "pdf_document_agent",
        "description": "Specialized in opening, reading, and extracting text from PDF documents. Can also find PDF files. Args: filepath (for reading), directory and pattern (for finding)."
    },
    AgentType.FILE_SEARCHING: {
        "tools": [
            find_files_by_pattern,
            list_directory_contents,
            grep_directory, 
            inspect_file_type_and_structure,
        ],
        "name": "file_system_search_agent",
        "description": "Specialized in searching for files by name or pattern, listing directory contents, and searching for text within files. Args: directory, patterns, search_terms."
    },
    AgentType.DATA_INSPECTING: {
        "tools": [
            inspect_file_type_and_structure,
            read_file_content, 
            list_directory_contents, 
            grep_directory,
        ],
        "name": "data_file_inspector_agent",
        "description": "Specialized in inspecting file types and structures (e.g., JSON, text), reading content from various files, and listing directories to locate data. Args: filepath or directory."
    },
    AgentType.PACKAGE_INSTALLING: {
        "tools": [
            install_python_package,
            check_python_package_version, 
            list_installed_python_packages,
            append_to_file,
        ],
        "name": "package_installing_agent",
        "description": "Specialized in installing packages that are needed (e.g., numpy, torch), checking installed packages, and figuring out dependencies, and adding it to the requirements.txt."
    },
    AgentType.FILE_MANAGING: {
        "tools": [
            create_directory,
            write_file, 
            delete_file_or_directory, 
            append_to_file, 
            list_directory_contents,
            inspect_file_type_and_structure,
            execute_shell_command,
            grep_directory,
            zip_files, 
            unzip_file,
            find_files_by_pattern
        ],
        "name": "file_managing_agent",
        "description": "Specialized in managing files and directories/folders, zipping/taring and unzipping/untaring, and figuring out the necessary and unnecessary files and folders, creating or deleting file or directory."
    },
    AgentType.AGENT_ORCHESTRATING: {
        "tools": [
            replace_text_in_file,
            execute_python_script,
            execute_shell_command,
            read_scratchpad, 
            update_scratchpad,
            inspect_file_type_and_structure,
        ],
        "name": "orchestrator",
        "description": "Director of the agents, does most  of the heavy lifting."
    }
}

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
            "additional_authorized_imports": [
                "os", "json", "sys", "collections", "glob", "shutil",
                "pandas", "numpy", "PIL", "matplotlib", "sklearn",
                "torch", "torchvision", "tensorflow", "logging" 
            ],
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
                                        # Google Gemini OpenAI-compatible API base URL
                                        api_base=kwargs.get("api_base"), #https://willma.liza.surf.nl/api/v0
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
        
        spec = AGENT_SPECIFICATIONS.get(agent_type)
        if not spec:
            raise ValueError(f"No specification found for agent type: {agent_type}")
        print("toools", spec["tools"])
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
        specialized_agent_strings: List[str],
        manager_agent_class: Type[CodeAgent] = CodeAgent,
        manager_name: str = "agent_orchestrating",
        manager_description: Optional[str] = None,
        manager_override_config: Optional[Dict[str, Any]] = None,
        specialized_agent_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
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
        if specialized_agent_configs is None:
            specialized_agent_configs = {}

        for agent_type_str in specialized_agent_strings:
            try:
                override_cfg = specialized_agent_configs.get(agent_type_str)
                agent = self.create_specialized_agent(agent_type_str, override_config=override_cfg)
                managed_agents.append(agent)
            except Exception as e:
                logger.error(f"Failed to create specialized agent of type '{agent_type_str}': {e}", exc_info=True)
                # Decide behavior: raise, or log and continue
                raise ValueError(f"Setup failed for specialized agent: {agent_type_str}") from e

        if manager_description is None:
            manager_description = (
                "I am a coordinator agent. I can delegate tasks to specialized agents "
                "for web Browse, PDF processing, file searching, and data inspection. "
                "Clearly state your high-level goal, and I will manage the sub-tasks."
            )
        agent_type = AgentType.from_string("agent_orchestrating")
        orchestrator_tools = AGENT_SPECIFICATIONS.get(agent_type)["tools"]

        self.agent = self._create_agent_instance(
            tools=orchestrator_tools,
            name=manager_name,
            description=manager_description,
            agent_class=manager_agent_class,
            managed_agents=managed_agents,
            override_config=manager_override_config
        )
        logger.info(f"Manager agent '{self.agent.name}' set up with {len(managed_agents)} specialized agents.")


    def setup_agent(
        self,
        list_of_tools: List[callable],
        agent_class: Type[MultiStepAgent] = CodeAgent,
        agent_config: Optional[Dict[str, Any]] = None,
        name: str = None,
        description: str = None,
        orchestrator: bool = False # Might not need this boolean
    ) -> CodeAgent:
        """
        Initializes and sets up an agent instance.

        Args:
            list_of_tools (List[callable]): A list of tool functions callable by the agent.
            agent_class (Type[MultiStepAgent]): The class of the agent to instantiate. Defaults to CodeAgent.
            agent_config (Dict[str, Any], optional): Configuration dictionary for the agent.
        """
        effective_agent_config = agent_config or {}
        agent_name = agent_class.__name__

        self.default_agent_configs["name"] = name
        self.default_agent_configs["description"] = description

        # Merge, with agent_config taking precedence
        merged_config = {**self.default_configs_configs, **effective_agent_config}

        try:
            agent = agent_class(
                tools=list_of_tools,
                model=self.model,
                **merged_config
            )

            if orchestrator:
                self.agent = agent

            model_id_str = self.model.model_id if hasattr(self.model, 'model_id') else 'N/A'
            logger.info(f"{agent_name} initialized successfully with {len(list_of_tools)} tools, model {model_id_str}, and config: {merged_config}")
        except Exception as e:
            logger.error(f"Error initializing {agent_name} with config {merged_config}: {e}", exc_info=True)
            raise RuntimeError(f"{agent_name} initialization failed: {e}")
        
        return agent

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