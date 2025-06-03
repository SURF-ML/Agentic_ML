import os
import logging
import argparse 
import yaml 
from typing import Dict, Any, List, Optional, Tuple

from utils.util_functions import load_config

from phases.ml_directive import Directive

from smolagents import CodeAgent
from agent_orchestrator import AgentOrchestrator

# Logger will be configured in main after parsing args for log level
logger = logging.getLogger(__name__)


def get_directives(prompt_details: dict, run_config: dict) -> List[str]:
    ml_directives = Directive(prompt_details)
    exec_phase = run_config.get("execute_phase")
    # get all the directives, but for now this is just one prompt for the main orchestrator
    directives = ml_directives.get_directives(exec_phase)[0]

    return directives

def setup_main_orchestrator(orchestrator: AgentOrchestrator,
                            config: dict) -> Tuple[AgentOrchestrator, str]:

    run_config = config.get('run', {})
    initial_prompt_filepath = run_config.get('initial_prompt_json', 'initial_project_config.json') 

    if orchestrator.load_from_json(initial_prompt_filepath):
        prompt_details = orchestrator.get_initial_prompt_details()
        logger.info(f"Loaded initial prompt details for project: {prompt_details.get('project_name', 'N/A') if prompt_details else 'N/A'}")
    else:
        logger.warning(f"Could not load initial prompt details from {initial_prompt_filepath}. Proceeding without them.")
        prompt_details = None

    # Where the agent will be working from
    work_dir = os.path.join(prompt_details.get("project_path"), prompt_details.get("project_name"))
    
    logger.info(f"Agent Model will be working from: {work_dir}")
    os.chdir(work_dir)

    agent_config = config.get('agent', {})

    directive = get_directives(prompt_details, run_config)
    logger.debug(f"Generated directive for agent:\n{directive}")
    #agent_list = agent_config.get("sub_agents")
    orchestrator_list = agent_config.get("orchestrator_agents")

    try:
        # TODO: all agents take the same (simple) config right now, fix this for individual agents?
        orchestrator.setup_main_orchestrator(
            orchestrator_agent_strings=orchestrator_list,
            orchestrator_configs=agent_config
        )

    except RuntimeError as e:
        logger.critical(f"Failed to set up agent via orchestrator: {e}", exc_info=True)
        return
    
    return orchestrator, directive

def run_main_orchestrator(orchestrator: AgentOrchestrator,
                          directive: str,) -> List[dict]:

    logger.info("Starting Agent Execution phase...")
    final_result = orchestrator.run_orchestrator(directive=directive)

    if isinstance(final_result, dict) and final_result.get("error"):
        logger.error(f"Agent phase execution failed: {final_result.get('message')}")
        logger.debug(f"Failure details: {final_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(final_result)[:1000]}")
    
    return final_result

def run_ml_agents(config: Dict[str, Any]):
    """
    Main function to set up and run the ML pipeline agent using AgentOrchestrator,
    driven by the loaded configuration.
    """
    logger.info("Starting ML Pipeline Agent setup with loaded configuration...")

    orchestrator = AgentOrchestrator(config)

    orchestrator, directive = setup_main_orchestrator(orchestrator, config)

    final_results = run_main_orchestrator(orchestrator, directive)
    
    last_result = final_results[-1]
    if isinstance(last_result, dict) and last_result.get("error"):
        logger.error(f"Agent phase execution failed: {last_result.get('message')}")
        logger.debug(f"Failure details: {last_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(last_result)[:1000]}")

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
    run_ml_agents(app_config)


if __name__ == "__main__":
    main()
