import os
import logging
import argparse
from typing import Dict, Any, List, Tuple

from agentic_ml.utils.util_functions import load_config, load_from_txt
from agentic_ml.directives.agency import Directive
from agentic_ml.orchestrator.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

def get_root_directive(template_path: str, initial_prompt: str) -> List[str]:
    """Generates the root directive from a template and an initial prompt."""
    directive = Directive(template_path)
    # get all the directives, but for now this is just one prompt for the main orchestrator
    directive = directive.get_directive(initial_prompt)
    return directive

def setup_root_orchestrator(orchestrator: AgentOrchestrator,
                            config: dict,
                            initial_prompt_override: str = None) -> Tuple[AgentOrchestrator, str]:
    """Sets up the root orchestrator with the necessary configuration and initial prompt."""
    run_config = config.get('run', {})
    prompt_details = ""

    # Use the command-line prompt if provided, otherwise load from file
    if initial_prompt_override:
        prompt_details = initial_prompt_override
        logger.info("Using initial prompt provided from command line.")
    else:
        initial_prompt_filepath = run_config.get('initial_prompt', 'initial_project.txt')
        prompt_details = load_from_txt(initial_prompt_filepath)
        logger.info(f"Loaded initial prompt details from {initial_prompt_filepath}")

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
        return None, None

    return orchestrator, directive

def run_root_orchestrator(orchestrator: AgentOrchestrator,
                          directive: str) -> List[dict]:
    """Runs the main orchestrator with the given directive."""
    logger.info("Starting Agent Execution phase...")
    final_result = orchestrator.run_orchestrator(directive=directive)

    if isinstance(final_result, dict) and final_result.get("error"):
        logger.error(f"Agent phase execution failed: {final_result.get('message')}")
        logger.debug(f"Failure details: {final_result.get('details')}")
    else:
        logger.info(f"Agent phase completed. Final output (or summary): {str(final_result)[:1000]}")

    return final_result

def run_agent(config: Dict[str, Any], initial_prompt_override: str = None):
    """
    Main function to set up and run the ML pipeline agent using AgentOrchestrator,
    driven by the loaded configuration and an optional prompt override.
    """
    logger.info("Starting ML Pipeline Agent setup with loaded configuration...")
    orchestrator = AgentOrchestrator(config)

    orchestrator, directive = setup_root_orchestrator(orchestrator, config, initial_prompt_override)

    if not orchestrator or not directive:
        logger.critical("Orchestrator setup failed. Aborting run.")
        return

    final_results = run_root_orchestrator(orchestrator, directive)

    if final_results:
        last_result = final_results[-1]
        if isinstance(last_result, dict) and last_result.get("error"):
            logger.error(f"Agent phase execution failed: {last_result.get('message')}")
            logger.debug(f"Failure details: {last_result.get('details')}")
        else:
            logger.info(f"Agent phase completed. Final output (or summary): {str(last_result)[:1000]}")

    logger.info("ML Pipeline Agent run finished.")

def main():
    """Parses command-line arguments and starts the agent run."""
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
    # New optional argument for the initial prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Directly provide the initial prompt, overriding the prompt file."
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
    logger.setLevel(numeric_level)

    try:
        config = load_config(args.config)
    except Exception:
        logger.critical(f"Failed to load configuration from {args.config}. Exiting.", exc_info=True)
        return

    # Run the agent, passing the configuration and the optional prompt override
    run_agent(config, args.prompt)


if __name__ == "__main__":
    main()