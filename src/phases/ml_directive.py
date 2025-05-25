import os
from typing import List, Dict

class Directive:

    def __init__(self, initial_prompt_details: Dict):
        self.root_path = "."
        self.scratchpad_file = initial_prompt_details.get('scratchpad_path', f"{os.path.join(self.root_path, 'agent/scratchpad.txt')}")

        self.initial_prompt = f"""Initial User Prompt:
            Project: {initial_prompt_details.get('project_name', 'N/A')}
            Task: {initial_prompt_details.get('task_description', 'N/A')}
            Data Type: {initial_prompt_details.get('data_type', 'N/A')}
            Raw Data Location: '{initial_prompt_details.get('data_folder_path_from_user', '../raw_data')}'
            Target Framework: {initial_prompt_details.get('target_framework', 'N/A')}
            Scratchpad path: {self.scratchpad_file}

            Assume at all times that you are working under "./",
            For instance, data/interim/ is found under {self.root_path}/data/interim and
            src/ is found under {self.root_path}/src, if you are unsure you can use `list_directory_contents`
            to inspect how your working directory looks like.

            If you encounter an unrecoverable error, are unsure about a step, or need clarification, use the `ask_user_for_input` tool.
            
            Use the scratchpad.txt, which is found under {self.root_path}/agent/scratchpad.txt to read detailed initial instructions and to log your progress, findings, and plans.
            Log as much as you can.

            If you are creating code, let the code be written as cleanly as possible. Use functions, keep the code modular, use typing and
            adhere to high software engineering standards.
            """

    def get_directives(self, directive: str="eda_preprocessing_directive") -> List[str]:
        return [self.inspect_and_understand_data_1(), 
                self.eda_preprocessing_directive_2()]
    
    def inspect_and_understand_data_1(self) -> str:
        """
        Constructs the detailed agent directive string for EDA and preprocessing.
        Args:
            initial_prompt_details: Dictionary containing the initial user prompt details.
        Returns:
            A string containing the agent directive.
        """

        return f"""
        {self.initial_prompt}

        Your current objective: Phase 1 - Understand Task and Data Inspection.

        Follow these general steps, using your available tools and reasoning capabilities. Finish this phase to the best
        of your ability! Keep in mind this is a high stakes project, everything you do will have impact further down the line.

        1.  **Understand Task & Locate Data:**
            * Add a task to `manage_agent_tasks` like "Understand task and locate data".
            * The Raw Data Location contains the data you'll work with.
            * Confirm this path. If it seems unusable, incorrect, or missing, first use `ask_user_for_input` to request clarification.
            * Log any understanding and stuff worth remembering using `update_scratchpad` tool.

        2.  **Initial Data Inspection:**
            * Add relevant tasks to `manage_agent_tasks` (e.g., "List project root directory", "Identify sample files", "Inspect sample files", "Summarize inspection findings").
            * Use `list_directory_contents` on the Project root path (recursive, max_depth 2-3).
            * Identify a few sample files. For each, use `inspect_file_type_and_structure`.
            * Summarize and log findings in the scratchpad.txt.

        3.  **Plan Preprocessing:**
            * Based on the previous results, plan the necessary preprocessing steps.
            * If the current findings are ambiguous for planning or if choices about preprocessing strategy are critical and uncertain, use `ask_user_for_input` to get user feedback or decisions.
            * Plan a manifest file (e.g., `{self.root_path}/manifests/phase_1/manifest.json`). 
            * Log your preprocessing plan in the scratchpad.txt.
            
        3.  **Conclude Phase 1:**
            * Ensure all planned tasks are "complete". If any are outstanding, address them or make a note in the scratchpad.txt about why they were not completed.
            * Summarize all actions taken, the current state of the data, and the preprocessing plan in the scratchpad.txt (`update_scratchpad`).
            * Indicate that Phase 1 (Understanding and Data Inspection) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Start by reading the scratchpad.txt and the agent_tasks.json for any existing notes and start by reading the manifest from the previous
        phase, if available, under {self.root_path}/manifests/phase_0/dataset_manifest.json, 
        It is imperative that you end the phase by logging and writing your finding in the {self.root_path}/manifests/phase_1/manifest.json`)!
        
        Inspect the {self.root_path}/src/ subdirectories and/or files, some default empty files have been set up for you. You can use these if you need.
        """

    def eda_preprocessing_directive_2(self) -> str:
        """
        Constructs the detailed agent directive string for EDA and preprocessing.
        Args:
            initial_prompt_details: Dictionary containing the initial user prompt details.
        Returns:
            A string containing the agent directive.
        """

        return f"""
        {self.initial_prompt}

        Your current objective: Phase 2 - EDA and Data Preprocessing.

        Follow these general steps, using your available tools and reasoning capabilities. Finish this phase to the best
        of your ability! Keep in mind this is a high stakes project, everything you do will have impact further down the line.
        You are at liberty to use as many tools as you want to perform the exploratory data analysis as professionally as possible.
        Use the web browsing tool if needed.

        1.  **Plan and Execute EDA:**
            * Based on `Data Type` and inspection, plan EDA. If the plan is unclear or you need to make choices with significant impact (e.g., how to handle a large amount of missing data).
            * Generate Python EDA script, save to `{self.root_path}/scripts/exploratory_data_analysis.py` using `write_file`.
                * The script should print summaries and save plots to `{self.root_path}/results/plots/eda/` (use `create_directory` if the path doesn't exist).
                * Optionally, the script can save an `eda_report.json` to `{self.root_path}/data/interim/`.
            * If you identify the need for new Python libraries (e.g., Pillow, pandas, matplotlib) for the EDA script:
                * First, use `read_file_content` to check if `requirements.txt` exists and if the libraries are already listed.
                * If not listed or the file doesn't exist, use `append_to_file` to add them to `{self.root_path}/requirements.txt`. Ensure each library is on a new line.
            * Execute EDA script using `execute_python_script`.
            * If the script execution fails:
                * Analyze the error output.
                * Attempt to debug common issues (e.g., incorrect paths, missing libraries you forgot to add).
                * If you cannot resolve it, use `ask_user_for_input` to present the error and ask for help or suggestions. E.g., "The EDA script failed with the following error: [...]. How should I proceed?".

        2.  **Plan Preprocessing:**
            * Based on EDA results and `Target Input Tensor Shape`, plan the necessary preprocessing steps.
            * If EDA findings are ambiguous for planning or if choices about preprocessing strategy are critical and uncertain, use `ask_user_for_input` to get user feedback or decisions.
            * Plan a manifest file (e.g., `{self.root_path}/manifests/phase_2/dataset_manifest.json`). 
            * Log your preprocessing plan in the scratchpad.txt.

        3.  **Conclude Phase 2:**
            * Ensure all planned tasks are "complete". If any are outstanding, address them or make a note in the scratchpad.txt about why they were not completed.
            * Summarize all actions taken, key findings from EDA, the current state of the data, and the preprocessing plan in the scratchpad.txt (`update_scratchpad`).
            * Indicate that Phase 2 (EDA and Data Preprocessing) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Start by reading the scratchpad.txt and the agent_tasks.json for any existing notes and start by reading the manifest from the previous phase under {self.root_path}/manifests/phase_1/manifest.json, 
        It is imperative that you end the phase by logging and writing your finding in the {self.root_path}/manifests/phase_2/manifest.json`)!
        """

# def define_scripting_testing_directive(initial_prompt_details: dict) -> str:
#     """
#     Constructs the detailed agent directive string for preprocessing script generation and testing.
#     Args:
#         initial_prompt_details: Dictionary containing the initial user prompt details.
#     Returns:
#         A string containing the agent directive.
#     """
#     scratchpad_file = initial_prompt_details.get('scratchpad_path', './agent_environment/agent/files/scratchpad.txt')
#     root_path = initial_prompt_details.get('project_path', 'N/A')
#     data_interim_path = f"{root_path}/data/interim"
#     data_processed_path = f"{root_path}/data/processed"
#     src_data_preprocessing_path = f"{root_path}/src/data_preprocessing"
#     tests_path = f"{root_path}/tests"

#     return f"""
#     Initial User Prompt Summary (Review):
#     Project: {initial_prompt_details.get('project_name', 'N/A')}
#     Project root path: {root_path}
#     Task: {initial_prompt_details.get('task_description', 'N/A')}
#     Data Type: {initial_prompt_details.get('data_type', 'N/A')}
#     Raw Data Location: '{initial_prompt_details.get('data_folder_path_from_user', 'N/A')}' # This was for Phase 1
#     Target Framework: {initial_prompt_details.get('target_framework', 'N/A')}
#     Target Input Tensor Shape: {initial_prompt_details.get('target_tensor_shapes_input', 'N/A')}
#     Scratchpad path: {scratchpad_file}

#     You have access to the following tools:
#     * `read_scratchpad`, `update_scratchpad`: For reading and writing in the scratchpad.
#     * `manage_agent_tasks`: For managing your to-do list (actions: "add", "remove", "list", "complete", "uncomplete", "clear").
#     * `ask_user_for_input`: To ask the human user for clarification, decisions, or information.
#     * `list_directory_contents`: To list contents of a directory.
#     * `inspect_file_type_and_structure`: To inspect file types and structures.
#     * `write_file`: To write content to a file.
#     * `read_file_content`: To read content from a file.
#     * `append_to_file`: To append content to a file.
#     * `execute_python_script`: To execute a Python script.
#     * `execute_shell_command`: To execute a shell command (e.g., for running tests).
#     * `create_directory`: To create a new directory.

#     Assume at all times that you are working under {root_path}.
#     For instance, '{data_processed_path}/' is the target for processed data.
#     '{src_data_preprocessing_path}/' is where preprocessing scripts should reside.
#     '{tests_path}/' is where test scripts should be placed.

#     If you encounter an unrecoverable error, are unsure about a step, or need clarification, use the `ask_user_for_input` tool.
#     Use `manage_agent_tasks` to add, track, and complete tasks corresponding to the steps below.

#     Your current objective: Phase 2 - Preprocessing Script Generation, Testing, and Finalization.
#     You should have completed Phase 1 (EDA and Data Preprocessing) and have a preprocessing plan in your scratchpad.

#     Follow these general steps:

#     1.  **Review Preprocessing Plan & Setup Tasks:**
#         * Add a task to `manage_agent_tasks` like "Review preprocessing plan and setup Phase 2 tasks".
#         * Read the scratchpad to refresh your understanding of the preprocessing plan established in Phase 1. This plan should guide the logic of your scripts.
#         * Identify the location of the data prepared in Phase 1 (likely in '{data_interim_path}'). This will be the input for your preprocessing scripts.
#         * Use `manage_agent_tasks` to "add" main tasks for this phase based on the steps below (e.g., "Generate Preprocessing Script", "Generate Tests", "Execute Preprocessing & Tests").
#         * Mark the setup task as "complete".

#     2.  **Generate Preprocessing Script(s):**
#         * Add a task to `manage_agent_tasks`: "Generate and save preprocessing script(s)".
#         * Based on the preprocessing plan from Phase 1:
#             * Generate Python code for preprocessing. You can choose to update an existing script like `{src_data_preprocessing_path}/preprocessing.py` or create a new one, for example, `{src_data_preprocessing_path}/build_processed_dataset.py`. If unsure about the script name or structure, `ask_user_for_input`.
#             * The script(s) must process data from the location identified in step 1 (e.g., '{data_interim_path}') and save the processed output to `{data_processed_path}/`. Ensure `{data_processed_path}` exists, using `create_directory` if necessary.
#             * **Crucially, ensure your preprocessing script also generates the `dataset_manifest.json` file as planned in Phase 1.** This manifest should detail the processed dataset (e.g., file paths, labels, dimensions). Save this manifest file to a suitable location, for instance, `{data_processed_path}/dataset_manifest.json` or `{data_interim_path}/dataset_manifest.json` (confirm from Phase 1 plan or decide and log). This file is critical for subsequent model training phases.
#         * Save the generated script(s) using `write_file`.
#         * If your script uses new libraries not previously listed, read `{root_path}/requirements.txt`, and if they are missing, append them using `append_to_file`.
#         * Log the script name, its purpose, input/output paths, and manifest file location in the scratchpad.
#         * Mark the task as "complete" in `manage_agent_tasks`.

#     3.  **Generate Tests for Preprocessing:**
#         * Add a task to `manage_agent_tasks`: "Generate and save unit tests for preprocessing".
#         * Generate Python unit tests for the preprocessing script(s) created in the previous step.
#         * Tests should cover key functionalities, data transformations, output structure, and the creation/format of `dataset_manifest.json`.
#         * Save the tests to `{tests_path}/test_data_preprocessing.py` using `write_file`. Ensure `{tests_path}` exists, using `create_directory` if needed.
#         * If necessary, create test fixtures (sample input data, expected output snippets) in `{tests_path}/fixtures/`. Use `create_directory` for the fixtures folder and `write_file` for fixture files.
#         * If you are unsure about specific test cases to cover, you can `ask_user_for_input` for suggestions (e.g., "What specific aspects of the {initial_prompt_details.get('data_type')} preprocessing should I prioritize for testing?").
#         * Log the name and location of the test script and any fixtures in the scratchpad.
#         * Mark the task as "complete" in `manage_agent_tasks`.

#     4.  **Execute Preprocessing & Tests:**
#         * Add tasks to `manage_agent_tasks`: "Execute preprocessing script" and "Execute unit tests".
#         * **Execute Preprocessing Script:**
#             * Run the main preprocessing script (identified in step 2) using `execute_python_script`.
#             * If the script fails: Analyze the error. Attempt to debug (e.g., check paths, library imports). If you cannot resolve it, use `ask_user_for_input` to present the error and ask for guidance.
#             * Once successful, log this in the scratchpad. Mark "Execute preprocessing script" as "complete".
#         * **Execute Unit Tests:**
#             * Run the unit tests (e.g., using `execute_shell_command` with a command like `python -m unittest {tests_path}/test_data_preprocessing.py` or `pytest {tests_path}`). If unsure about the exact command, you can `ask_user_for_input`.
#             * If tests fail: Analyze the failure. Attempt to fix the preprocessing script or the tests accordingly. This might involve re-running the preprocessing script if it was changed. If you cannot resolve failures, use `ask_user_for_input` for help.
#             * Once all tests pass, log this in the scratchpad. Mark "Execute unit tests" as "complete".
#         * **Verify Outputs:**
#             * Add a task "Verify preprocessing outputs".
#             * Use `list_directory_contents` to inspect `{data_processed_path}/` and confirm that processed files are present.
#             * Specifically, verify that `dataset_manifest.json` has been created in the expected location and inspect its content briefly (e.g., using `read_file_content` to check for expected keys or structure).
#             * Log your verification findings in the scratchpad.
#             * Mark "Verify preprocessing outputs" as "complete".

#     5.  **Conclude Phase 2:**
#         * Use `manage_agent_tasks` with the "list" action to ensure all planned tasks for Phase 2 are "complete". If not, address them or use `ask_user_for_input` if guidance is needed.
#         * Summarize actions taken, scripts created, test results, and the state of processed data (including the manifest file) in the Scratchpad.
#         * Indicate that Phase 2 is complete and you are ready for the next phase (e.g., model training).

#     Start by reading the scratchpad for the preprocessing plan from Phase 1 and then proceed with step 1 of this directive.
#     Always provide clear reasoning for your actions and use the available tools methodically.
#     """