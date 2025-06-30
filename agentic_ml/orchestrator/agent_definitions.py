from enum import Enum

from agentic_ml.agent.tools import (create_directory, write_file, append_to_file, read_file_content,
    list_directory_contents, delete_file_or_directory, replace_text_in_file,
    execute_python_script, execute_shell_command, install_python_package,
    browse_webpage, search_arxiv, search_github_repositories, inspect_file_type_and_structure,
    search_wikipedia, ask_user_for_input, check_python_package_version, 
    list_installed_python_packages, grep_directory, zip_files, unzip_file,
    search_google_scholar, download_file_from_url, read_pdf_content,
    find_files_by_pattern, spawn_and_run_agent

)

from smolagents import DuckDuckGoSearchTool, WebSearchTool, VisitWebpageTool, WikipediaSearchTool#, ApiWebSearchTool

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
    ORCHESTRATOR = "orchestrator"
    BROWSER = "browser"
    FILE_NAVIGATOR = "file_navigator"

    @classmethod
    def from_string(cls, s: str) -> 'AgentType':
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"Unknown agent type string: {s}")

AGENT_TASK = {

    AgentType.ORCHESTRATOR: {
        "tools": [
            create_directory, 
            write_file, 
            append_to_file, 
            read_file_content,
            delete_file_or_directory, 
            replace_text_in_file,
            execute_python_script, 
            execute_shell_command, 
            install_python_package,
            ask_user_for_input, 
            check_python_package_version, 
            list_installed_python_packages, 
            zip_files, 
            unzip_file,
            read_pdf_content,
            spawn_and_run_agent
        ],
        "name": "orchestrator",
        "description": "Director of the agents, does most of the heavy lifting."
    },
    AgentType.BROWSER: {
        "tools": [
            DuckDuckGoSearchTool(), 
            #ApiWebSearchTool(), 
            #WebSearchTool(), 
            VisitWebpageTool(), 
            WikipediaSearchTool(),
            search_arxiv,
            search_github_repositories,
            search_google_scholar,
            download_file_from_url,
        ],
        "name": "browser",
        "description": "Specialized in web navigation, using online search engines (Wikipedia, arXiv, GitHub, Google Scholar), and downloading files from URLs. Args: query (for searches) or url (for Browse/downloading)."
    },
    AgentType.FILE_NAVIGATOR: {
        "tools": [
            find_files_by_pattern,
            list_directory_contents,
            grep_directory, 
            inspect_file_type_and_structure,
        ],
        "name": "file_navigator",
        "description": "Specialized in searching for files by name or pattern, listing directory contents, and searching for text within files. Args: directory, patterns, search_terms."
    },
}