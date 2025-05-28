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
    # TODO: function should get list of strings as input
    def get_directives(self, directive_key: str = "all") -> List[str]:
        """
        Returns a list of directive strings based on the provided key.
        Args:
            directive_key: Specifies which directives to return.
                           "all": returns all defined directives in sequence.
                           "phase_1": returns inspect_and_understand_data_1.
                           "phase_2": returns eda_preprocessing_directive_2.
                           "phase_3": returns data_preprocessing_and_feature_engineering_3.
                           "phase_4": returns model_selection_and_initial_training_4.
                           "phase_5": returns model_evaluation_and_iteration_planning_5.
                           Default is "all".
        Returns:
            A list of strings, where each string is a directive for a phase.
        """
        directives_map = {
            "phase_1": [self.inspect_and_understand_data_1()],
            "phase_2": [self.eda_preprocessing_directive_2()],
            "phase_3": [self.data_preprocessing_and_feature_engineering_3()],
            "phase_4": [self.model_selection_and_initial_training_4()],
            #"phase_5": [self.model_evaluation_and_iteration_planning_5()],
        }
        if directive_key == "all":
            return [
                self.inspect_and_understand_data_1(),
                self.eda_preprocessing_directive_2(),
                self.data_preprocessing_and_feature_engineering_3(),
                self.model_selection_and_initial_training_4(),
                #self.model_evaluation_and_iteration_planning_5(),
            ]
        return directives_map.get(directive_key, [f"Error: Unknown directive key '{directive_key}'."])
    
    
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
            * The Raw Data Location contains the data you'll work with.
            * Confirm this path. If it seems unusable, incorrect, or missing, first use `ask_user_for_input` to request clarification.
            * Log any understanding and stuff worth remembering using `update_scratchpad` tool.

        2.  **Initial Data Inspection:**
            * Use `list_directory_contents` on the Project root path (recursive, max_depth 2-3).
            * Identify a few sample files. For each, use `inspect_file_type_and_structure`.
            * Summarize and log findings in the scratchpad.txt.

        3.  **Plan Preprocessing:**
            * Based on the previous results, plan the necessary preprocessing steps.
            * If the current findings are ambiguous for planning or if choices about preprocessing strategy are critical and uncertain, use `ask_user_for_input` to get user feedback or decisions.
            * Plan a manifest file (e.g., `{self.root_path}/manifests/phase_1/manifest.json`). Add as much information as you can. 
            * Log your preprocessing plan in the scratchpad.txt.
            
        3.  **Conclude Phase 1:**
            * Ensure all planned tasks are "complete". If any are outstanding, address them or make a note in the scratchpad.txt about why they were not completed.
            * Summarize all actions taken, the current state of the data, and the preprocessing plan in the scratchpad.txt (`update_scratchpad`).
            * Indicate that Phase 1 (Understanding and Data Inspection) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Start by reading the scratchpad.txt and the manifests for any existing notes, if available, under {self.root_path}/manifests/phase_0/dataset_manifest.json, 
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
            * Plan a manifest file (e.g., `{self.root_path}/manifests/phase_2/dataset_manifest.json`). Add as much information as you can.
            * Log your preprocessing plan in the scratchpad.txt.

        3.  **Conclude Phase 2:**
            * Ensure all planned tasks are "complete". If any are outstanding, address them or make a note in the scratchpad.txt about why they were not completed.
            * Summarize all actions taken, key findings from EDA, the current state of the data, and the preprocessing plan in the scratchpad.txt (`update_scratchpad`).
            * Indicate that Phase 2 (EDA and Data Preprocessing) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Start by reading the scratchpad.txt for any existing notes and start by reading the manifest from the previous phase under {self.root_path}/manifests/phase_1/manifest.json, 
        It is imperative that you end the phase by logging and writing your finding in the {self.root_path}/manifests/phase_2/manifest.json`)!
        """

    def data_preprocessing_and_feature_engineering_3(self) -> str:
        """
        Constructs the detailed agent directive string for implementing data preprocessing and feature engineering.
        """
        return f"""
        {self.initial_prompt}

        Your current objective: Phase 3 - Data Preprocessing Implementation and Feature Engineering.

        Follow these general steps, using your available tools and reasoning capabilities.
        Start by reading {self.scratchpad_file}, and the manifest from Phase 2: `{self.root_path}/manifests/phase_2/manifest.json`.

        1.  **Implement Preprocessing Script:**
            * Add tasks: "Develop data preprocessing script", "Execute preprocessing script", "Verify processed data".
            * Based on the detailed plan in the Phase 2 manifest, write a Python script (e.g., `{self.root_path}/scripts/data_preprocessing.py`) using `write_file`.
                * This script should load raw or interim data (as identified in previous phases).
                * Apply all planned cleaning, transformation, encoding, and scaling steps.
                * Save the processed data to a designated output directory (e.g., `{self.root_path}/data/processed/`). Use `create_directory` for this path. Ensure output format is suitable for model training (e.g., CSV, Parquet, NumPy arrays).
            * Check and update `{self.root_path}/requirements.txt` using `read_file_content` and `append_to_file` if new libraries are introduced for preprocessing.
            * Execute the script using `execute_python_script`. Analyze output/errors. Debug or use `ask_user_for_input` if issues arise.
            * Log script location, processed data path, and any important observations in {self.scratchpad_file}.

        2.  **Plan and Implement Feature Engineering (if applicable):**
            * Add task: "Plan and implement feature engineering".
            * Review EDA findings and the `Project Task`. Identify opportunities for feature engineering (e.g., creating interaction terms, polynomial features, extracting features from text/date, dimensionality reduction like PCA).
            * If not already part of the preprocessing script, you might create a separate script (e.g., `{self.root_path}/scripts/feature_engineering.py`) or modify the existing one. Use `write_file`.
            * If the strategy for feature engineering is complex or has many alternatives, briefly outline your proposal in the scratchpad and use `ask_user_for_input` for confirmation or guidance.
            * Implement the chosen feature engineering steps. This might involve re-running parts of your preprocessing script or executing a new one.
            * Ensure engineered features are saved alongside or as part of the processed dataset.
            * Log the feature engineering plan, implementation details, and outcomes in {self.scratchpad_file}.

        3.  **Verify Processed Data Quality:**
            * Add task: "Verify quality and structure of processed data".
            * Perform a quick inspection of the processed data. Use `inspect_file_type_and_structure` or write a small script to load and check shape, data types, and a sample of the processed data.
            * Confirm it aligns with expectations and the `Target Input Tensor Shape` (if provided).
            * Log verification results in {self.scratchpad_file}.

        4.  **Conclude Phase 3:**
            * Summarize actions, details of implemented preprocessing and feature engineering, and final processed data characteristics in {self.scratchpad_file}.
            * Create the manifest: `{self.root_path}/manifests/phase_3/processed_data_manifest.json`. This file should detail:
                * Path to the preprocessing/feature engineering script(s).
                * Description of all transformations and features created.
                * Path to the final processed dataset(s).
                * Shape and data types of the processed data.
                * Any relevant statistics about the processed data (e.g., number of features, samples).
            * Use `write_file` to create this manifest.
            * Indicate in {self.scratchpad_file} that Phase 3 is complete.

        Remember to maintain clean, modular code in your scripts.
        """
    
    def model_selection_and_initial_training_4(self) -> str:
        """
        Constructs the detailed agent directive string for model selection and initial training.
        """
        return f"""
        {self.initial_prompt}

        Your current objective: Phase 4 - Model Selection and Initial Training.

        Follow these general steps.
        Start by reading {self.scratchpad_file} and the manifest from Phase 3: `{self.root_path}/manifests/phase_3/manifest.json`.

        1.  **Review Processed Data and Task for Model Selection:**
            * Load information about the processed data from the Phase 3 manifest.
            * Based on this, research suitable machine learning models. Use `search_arxiv`, `search_google_scholar`, or `search_github_repositories` for ideas or implementations if needed.
            * This is the most important step for a successful completion of this phase, make sure to use the scratchpad to write down your thoughts and research which model is the best one for this task.
            * Select a candidate model appropriate for the task. Justify your choices in {self.scratchpad_file}. If uncertain about choices, use `ask_user_for_input`.

        2.  **Develop Training Script(s):**
            * Create a Python script (e.g., `{self.root_path}/src/model_training.py`. Use `write_file`.
            * The script should:
                * Load the processed data (path from Phase 3 manifest).
                * Implement data splitting (e.g., train, validation, test sets). Ensure this is done consistently.
                * Define and implement the selected model architectures using the `Target Framework`.
                * Set up the training loop, including loss function, optimizer, and relevant metrics.
                * Include functionality to save trained model artifacts (e.g., to `{self.root_path}/models/model_name/`) and training logs/metrics (e.g., to `{self.root_path}/results/training_logs/`). Use `create_directory` for these paths.
            * Check and update `{self.root_path}/requirements.txt` for any new libraries (e.g., scikit-learn, torch). Use `read_file_content`, `append_to_file`.
            * Use functions, typing and classes. Make sure everything is modularized.

        3.  **Perform Initial Training Runs:**
            * Add task: "Execute initial training for candidate models".
            * Execute the training script for each candidate model. Use `execute_python_script`.
            * Monitor the training process (if possible, through script output).
            * If training fails or produces unexpected errors, analyze the output, debug the script, or use `ask_user_for_input` for assistance.
            * Log paths to saved models, training logs, and key performance metrics (e.g., training/validation loss and accuracy) in {self.scratchpad_file}. Use `log_agent_message` for detailed logs if useful.

        4.  **Conclude Phase 4:**
            * Summarize actions, selected models, training setup, and initial training results in {self.scratchpad_file}.
            * Create the manifest: `{self.root_path}/manifests/phase_4/manifest.json`. This JSON file should include:
                * List of models trained.
                * Path to the training script(s).
                * Details of data splits.
                * Paths to saved model artifacts for each model.
                * Key initial training/validation metrics for each model.
                * Path to any training log files.
            * Use `write_file` to create this manifest.
            * Indicate in {self.scratchpad_file} that Phase 4 is complete.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Browse the web for inspiration on which type of model or method should be best suited for this type of task, work to the best of your ability to improve the model.
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