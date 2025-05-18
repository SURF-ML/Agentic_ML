import logging
import argparse 
import yaml 
from typing import Dict, Any, List, Optional

# --- Custom Tools Import ---
from agent_environment.agent.tools import (
    create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, search_arxiv, search_github_repositories,
    read_scratchpad, update_scratchpad, inspect_file_type_and_structure,
    search_wikipedia, log_agent_message, list_agent_log_files, read_scratchpad,
    update_scratchpad
)
ALL_CUSTOM_TOOLS: List[callable] = [
    create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, search_arxiv, search_github_repositories,
    read_scratchpad, update_scratchpad, inspect_file_type_and_structure,
    search_wikipedia, log_agent_message, list_agent_log_files
]
DATA_PHASE_TOOLS: List[callable] = [
    create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, read_scratchpad, update_scratchpad, inspect_file_type_and_structure,
    log_agent_message, list_agent_log_files
]

from phases.data_phase.data_objective import define_eda_preprocessing_directive

from smolagents import CodeAgent

from agent_orchestrator import AgentOrchestrator

# Logger will be configured in main after parsing args for log level
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config from {config_path}: {e}")
        raise


# --- Directive Generation (Example) ---
def generate_initial_directive(prompt_details: Optional[Dict[str, Any]], user_query: str) -> str:
    """
    Generates a directive for the agent based on initial details and a user query.
    """
    # This function remains the same as before
    if prompt_details:
        directive = f"""
        Project Context (loaded from initial prompt details):
        Project Name: {prompt_details.get('project_name', 'N/A')}
        Task Description: {prompt_details.get('task_description', 'N/A')}
        Data Type: {prompt_details.get('data_type', 'N/A')}
        Target Framework: {prompt_details.get('target_framework', 'N/A')}

        User's Current Request:
        {user_query}

        Please proceed with the user's request, keeping the project context in mind.
        Use your available tools and reasoning capabilities.
        Log your major steps and findings.
        """
    else:
        directive = f"""
        User's Current Request:
        {user_query}

        Please proceed with the user's request using your available tools and reasoning capabilities.
        Log your major steps and findings.
        """
    return directive.strip()

def run_data_phase(orchestrator: AgentOrchestrator, 
                   final_agent_config: dict, 
                   directive: str,
                   initial_prompt_file: dict) -> CodeAgent:
    try:
        orchestrator.setup_agent(
            list_of_tools=DATA_PHASE_TOOLS,
            agent_class=CodeAgent,
            agent_config=final_agent_config
        )
    except RuntimeError as e:
        logger.critical(f"Failed to set up agent via orchestrator: {e}", exc_info=True)
        return

    directive = f"directive\n\n{define_eda_preprocessing_directive(initial_prompt_file)}"

    logger.debug(f"Generated directive for agent:\n{directive}")

    logger.info("Starting agent execution phase...")
    final_result = orchestrator.run_agent_phase(directive=directive)

    if isinstance(final_result, dict) and final_result.get("error"):
        logger.error(f"Agent phase execution failed: {final_result.get('message')}")
        logger.debug(f"Failure details: {final_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(final_result)[:1000]}")


def run_ml_pipeline_agent(config: Dict[str, Any]):
    """
    Main function to set up and run the ML pipeline agent using AgentOrchestrator,
    driven by the loaded configuration.
    """
    logger.info("Starting ML Pipeline Agent setup with loaded configuration...")

    orchestrator = AgentOrchestrator(config)

    paths_config = config.get('paths', {})
    initial_prompt_filepath = paths_config.get('initial_prompt_json', 'initial_project_config.json') 

    if orchestrator.load_initial_prompt_from_json(initial_prompt_filepath):
        prompt_details = orchestrator.get_initial_prompt_details()
        logger.info(f"Loaded initial prompt details for project: {prompt_details.get('project_name', 'N/A') if prompt_details else 'N/A'}")
    else:
        logger.warning(f"Could not load initial prompt details from {initial_prompt_filepath}. Proceeding without them.")
        prompt_details = None

    agent_config_from_yaml = config.get('agent', {})
    # Default agent settings, can be overridden by YAML
    default_agent_settings = {
        "stream_outputs": True,
        "max_steps": 30,
        "additional_authorized_imports": [
            "torch", "numpy", "pandas", "sklearn", "matplotlib", "PIL",
            "os", "json", "sys", "glob", "shutil", "logging"
        ]
    }
    # Merge default with YAML config, YAML takes precedence
    final_agent_config = {**default_agent_settings, **agent_config_from_yaml}

    #Define the User's Query / Initial Task for the Agent
    user_query = prompt_details["task_description"]

    directive = generate_initial_directive(prompt_details, user_query)
    run_data_phase(orchestrator, final_agent_config, directive, prompt_details)

    logger.info("Starting agent execution phase...")
    final_result = orchestrator.run_agent_phase(directive=directive)

    if isinstance(final_result, dict) and final_result.get("error"):
        logger.error(f"Agent phase execution failed: {final_result.get('message')}")
        logger.debug(f"Failure details: {final_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(final_result)[:1000]}")

    logger.info("ML Pipeline Agent run finished.")

def main():
    parser = argparse.ArgumentParser(description="Run the ML Pipeline Agent with a specified configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml", 
        help="Path to the YAML configuration file for the agent pipeline."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    args = parser.parse_args()

    # Configure logging level based on command line argument
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    
    # Setup root logger configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    # Re-initialize module logger with new level if basicConfig was called
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level) # Ensure this specific logger also respects the level

    try:
        app_config = load_config(args.config)
    except Exception:
        logger.critical(f"Failed to load configuration from {args.config}. Exiting.")
        return

    # Run the pipeline with the loaded config
    run_ml_pipeline_agent(app_config)


if __name__ == "__main__":
    main()
