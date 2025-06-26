from enum import Enum

from agentic_ml.agent.tools import (create_directory, write_file, append_to_file, read_file_content,
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