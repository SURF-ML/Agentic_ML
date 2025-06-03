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
    # Single Task Agents
    BROWSING = "browsing"
    PDF_OPENING = "pdf_opening"
    FILE_SEARCHING = "file_searching"
    DATA_INSPECTING = "data_inspecting"
    PACKAGE_INSTALLING = "package_installing"
    FILE_MANAGING = "file_managing"
    AGENT_ORCHESTRATING = "agent_orchestrating"
    
    # Orchestrator agents
    Research_Orchestrator = "research_orchestrator"
    EnvironmentSetup_Orchestrator = "environmentsetup_orchestrator"
    DataIngestionValidation_Orchestrator = "dataingestionvalidation_orchestrator"
    ExploratoryDataAnalysis_Orchestrator = "exploratorydataanalysis_orchestrator"
    DataPreprocessingFeatureEngineering_Orchestrator = "datapreprocessingfeatureengineering_orchestrator"
    ModelTraining_Orchestrator = "modeltraining_orchestrator"
    HyperparameterOptimization_Orchestrator = "hyperparameteroptimization_orchestrator"
    ModelEvaluation_Orchestrator = "modelevaluation_orchestrator"
    Reporting_Orchestrator = "reporting_orchestrator"
    
    @classmethod
    def from_string(cls, s: str) -> 'AgentType':
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown agent type string: {s}")

AGENT_SINGLE_TASK = {

    AgentType.BROWSING: {
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
            ask_user_for_input
        ],
        "name": "orchestrator",
        "description": "Director of the agents, does most of the heavy lifting."
    }
}

AGENT_ORCHESTRATOR = {
    AgentType.Research_Orchestrator: {
        "managed_agents": [
            "browsing", 
            "pdf_opening", 
            "file_searching", 
            "file_managing"
        ],
        "name": "research_orchestrator",
        "description": "Scours the internet (general web, academic papers, code repositories) to develop a creative, high-level plan for tackling a novel or complex problem. It synthesizes its findings into a proposed strategy with clear, actionable steps. Worker agents it spawns: web_navigator_agent, pdf_document_agent, file_system_search_agent, file_managing_agent. Core tools: update_scratchpad, write_file."
    },
    AgentType.EnvironmentSetup_Orchestrator: {
        "managed_agents": [
            "package_installing",
            "file_managing"
        ],
        "name": "environmentsetup_orchestrator",
        "description": "Ensures the project environment is correctly set up. Installs required base packages and creates the necessary directory structure for the project. Worker agents it spawns: package_installing_agent, file_managing_agent. Core tools: execute_shell_command, update_scratchpad."
    },
    AgentType.DataIngestionValidation_Orchestrator: {
        "managed_agents": [
            "browsing", 
            "pdf_opening", 
            "file_searching", 
            "data_inspecting", 
            "file_managing"
        ],
        "name": "dataingestionvalidation_orchestrator",
        "description": "Manages the entire data acquisition and validation process. Finds, downloads, and inspects data from web or local sources, checks for quality and consistency, and prepares a preliminary data report. Worker agents it spawns: web_navigator_agent, pdf_document_agent, file_system_search_agent, data_file_inspector_agent, file_managing_agent. Core tools: execute_python_script, read_scratchpad, update_scratchpad."
    },
    AgentType.ExploratoryDataAnalysis_Orchestrator: {
        "managed_agents": [
            "data_inspecting",
            "file_managing",
            "package_installing"
        ],
        "name": "exploratorydataanalysis_orchestrator",
        "description": "Performs a deep dive into the validated data. Its goal is to understand data distributions, find correlations, identify anomalies, and generate summary statistics and visualizations to inform the next steps. Worker agents it spawns: data_file_inspector_agent, file_managing_agent, package_installing_agent. Core tools: execute_python_script, read_scratchpad, update_scratchpad."
    },
    AgentType.DataPreprocessingFeatureEngineering_Orchestrator: {
        "managed_agents": [
            "file_managing",
            "data_inspecting",
            "package_installing"
        ],
        "name": "datapreprocessingfeatureengineering_orchestrator",
        "description": "Transforms the raw data into a clean, model-ready format. It handles tasks like normalization, scaling, encoding, and imputation. Crucially, it also generates new features based on EDA insights. Worker agents it spawns: file_managing_agent, data_file_inspector_agent, package_installing_agent. Core tools: execute_python_script, read_scratchpad, update_scratchpad."
    },
    AgentType.ModelTraining_Orchestrator: {
        "managed_agents": [
            "file_managing",
            "package_installing",
            "data_inspecting"
        ],
        "name": "modeltraining_orchestrator",
        "description": "Selects appropriate model architectures and trains them on the preprocessed data. It logs training parameters and initial performance metrics, saving the trained model artifacts. Worker agents it spawns: file_managing_agent, package_installing_agent. Core tools: execute_python_script, execute_shell_command, read_scratchpad, update_scratchpad."
    },
    AgentType.HyperparameterOptimization_Orchestrator: {
        "managed_agents": [
            "file_managing",
            "data_inspecting"
        ],
        "name": "hyperparameteroptimization_orchestrator",
        "description": "Systematically tunes the hyperparameters of a trained model to maximize performance. It can employ strategies like grid search, random search, or more advanced methods, often using cross-validation. Worker agents it spawns: file_managing_agent. Core tools: execute_python_script, read_scratchpad, update_scratchpad."
    },
    AgentType.ModelEvaluation_Orchestrator: {
        "managed_agents": [
            "data_inspecting",
            "file_managing"
        ],
        "name": "modelevaluation_orchestrator",
        "description": "Conducts a rigorous evaluation of the final, tuned model on a hold-out test set. It calculates a comprehensive set of performance metrics and generates comparison tables if multiple models were trained. Worker agents it spawns: data_file_inspector_agent, file_managing_agent. Core tools: execute_python_script, read_scratchpad, update_scratchpad."
    },
    AgentType.Reporting_Orchestrator: {
        "managed_agents": [
            "browsing",
            "file_searching",
            "data_inspecting"
        ],
        "name": "reporting_orchestrator",
        "description": "Synthesizes the entire project lifecycle or a complex phase into a final, human-readable report. It gathers results, code, and insights from the scratchpad and project files to generate a summary document. Worker agents it spawns: web_navigator_agent, file_system_search_agent, data_file_inspector_agent. Core tools: execute_python_script, read_file_content, write_file."
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