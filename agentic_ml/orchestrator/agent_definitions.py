from enum import Enum

from agentic_ml.agent.tools import (create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, search_arxiv, search_github_repositories,
    read_scratchpad, update_scratchpad, inspect_file_type_and_structure,
    search_wikipedia, ask_user_for_input, check_python_package_version, 
    list_installed_python_packages, grep_directory, zip_files, unzip_file,
    search_google_scholar, download_file_from_url, read_pdf_content,
    find_files_by_pattern, spawn_and_run_agent

)

ALL_TOOLS = [create_directory, 
            write_file, 
            append_to_file, 
            read_file_content,
            list_directory_contents, 
            delete_file_or_directory, 
            replace_text_in_file,
            execute_python_script, 
            execute_shell_command, 
            install_python_package,
            browse_webpage, 
            search_arxiv, 
            search_github_repositories,
            read_scratchpad, 
            update_scratchpad, 
            inspect_file_type_and_structure,
            search_wikipedia, 
            ask_user_for_input, 
            check_python_package_version, 
            list_installed_python_packages, 
            grep_directory, 
            zip_files, 
            unzip_file,
            search_google_scholar, 
            download_file_from_url, 
            read_pdf_content,
            find_files_by_pattern,
            spawn_and_run_agent]

class AgentType(Enum):
   
    # Orchestrator agents
    AGENT_ORCHESTRATOR = "agent_orchestrator"

    @classmethod
    def from_string(cls, s: str) -> 'AgentType':
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown agent type string: {s}")

AGENT_SINGLE_TASK = {

    AgentType.AGENT_ORCHESTRATOR: {
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