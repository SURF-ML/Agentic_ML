import os
import logging
import argparse 
from typing import Dict, Any, List, Tuple

from agentic_ml.utils.util_functions import load_config, load_from_txt
from agentic_ml.directives.agency import Directive
from agentic_ml.orchestrator.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

def get_root_directive(template_path: str, initial_prompt: str) -> List[str]:
    directive = Directive(template_path)
    # get all the directives, but for now this is just one prompt for the main orchestrator
    directive = directive.get_directive(initial_prompt)

    return directive

def setup_root_orchestrator(orchestrator: AgentOrchestrator,
                            config: dict) -> Tuple[AgentOrchestrator, str]:

    run_config = config.get('run', {})
    initial_prompt_filepath = run_config.get('initial_prompt', 'initial_project.txt') 

    prompt_details = load_from_txt(initial_prompt_filepath)
    logger.info(f"Loaded initial prompt details for project")
    
    logger.info(f"Agent Model will be working from: {run_config.get('agent_working_dir')}")
    os.chdir(run_config.get('agent_working_dir'))

    agent_config = config.get('agent', {})

    directive = get_root_directive(agent_config.get("template_path"), prompt_details)
    logger.debug(f"Generated directive for agent:\n{directive}")

    try:
        # TODO: all agents take the same (simple) config right now, fix this for individual agents?
        orchestrator.setup_main_orchestrator(
            orchestrator_configs=agent_config
        )

    except RuntimeError as e:
        logger.critical(f"Failed to set up agent via orchestrator: {e}", exc_info=True)
        return
    
    return orchestrator, directive

def run_root_orchestrator(orchestrator: AgentOrchestrator,
                          directive: str,) -> List[dict]:

    logger.info("Starting Agent Execution phase...")
    final_result = orchestrator.run_orchestrator(directive=directive)

    if isinstance(final_result, dict) and final_result.get("error"):
        logger.error(f"Agent phase execution failed: {final_result.get('message')}")
        logger.debug(f"Failure details: {final_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(final_result)[:1000]}")
    
    return final_result

def run_agent(config: Dict[str, Any]):
    """
    Main function to set up and run the ML pipeline agent using AgentOrchestrator,
    driven by the loaded configuration.
    """
    logger.info("Starting ML Pipeline Agent setup with loaded configuration...")

    orchestrator = AgentOrchestrator(config)

    orchestrator, directive = setup_root_orchestrator(orchestrator, config)

    final_results = run_root_orchestrator(orchestrator, directive)
    
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
        config = load_config(args.config)
    except Exception:
        logger.critical(f"Failed to load configuration from {args.config}. Exiting.")
        return

    # Run the pipeline with the loaded config
    run_agent(config)


if __name__ == "__main__":
    main()
