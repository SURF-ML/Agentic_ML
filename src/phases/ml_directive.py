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
            src/ is found under {self.root_path}/src, if you are unsure you can use the `file_system_search_agent`
            to inspect how your working directory looks like.

            If you encounter an unrecoverable error, are unsure about a step, or need clarification, use the `ask_user_for_input` tool.
            
            Use the scratchpad.txt, which is found under {self.root_path}/agent/scratchpad.txt.
            You can use the `data_file_inspector_agent` to read detailed initial instructions from it.
            Log your progress, findings, and plans directly to it (e.g. using the `update_scratchpad` tool or a similar direct logging capability).
            Log as much as you can.

            If you are creating code, let the code be written as cleanly as possible. Use functions, keep the code modular, use typing and
            adhere to high software engineering standards. Log as much as possible in the scratchpad, this will be used to verify and check what you have done.
            Write very detailed account in the manifests and in the scratchpad. Keep using the `file_system_search_agent` to understand how you are changing
            the directories and files. Always start by inspecting the directories and their contents recursively; you need to know if there are any files
            that you can use. Be creative and try to do max effort, your attempt will be graded!
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
                           # "phase_5": returns model_evaluation_and_iteration_planning_5.
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

        Follow these general steps, using your available sub-agents and reasoning capabilities. Finish this phase to the best
        of your ability! Keep in mind this is a high stakes project, everything you do will have impact further down the line.

        1.  **Understand Task & Locate Data:**
            * The Raw Data Location contains the data you'll work with.
            * Confirm this path. If it seems unusable, incorrect, or missing, first use `ask_user_for_input` to request clarification.
            * Log any understanding and stuff worth remembering in the scratchpad.txt (e.g., using the `update_scratchpad` tool).

        2.  **Initial Data Inspection:**
            * Use the `file_system_search_agent` to list contents of the Project root path (recursive, max_depth 2-3).
            * Identify a few sample files. For each, use the `data_file_inspector_agent` to inspect its type and structure.
            * Summarize and log findings in the scratchpad.txt.

        3.  **Plan Preprocessing:**
            * Based on the previous results, plan the necessary preprocessing steps.
            * If the current findings are ambiguous for planning or if choices about preprocessing strategy are critical and uncertain, use `ask_user_for_input` to get user feedback or decisions.
            * Plan a manifest file (e.g., `{self.root_path}/manifests/phase_1/manifest.json`). Add as much information as you can. 
            * Log your preprocessing plan in the scratchpad.txt.
            
        4.  **Conclude Phase 1:**
            * Ensure all planned tasks are "complete". If any are outstanding, address them or make a note in the scratchpad.txt about why they were not completed.
            * Summarize all actions taken, the current state of the data, and the preprocessing plan in the scratchpad.txt.
            * Indicate that Phase 1 (Understanding and Data Inspection) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging of your progress, observations, and plans.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors you cannot resolve, or need a decision that impacts the workflow.
        Start by using the `data_file_inspector_agent` to read the scratchpad.txt and any manifests for existing notes, if available, under `{self.root_path}/manifests/phase_0/dataset_manifest.json`.
        It is imperative that you end the phase by using the `file_managing_agent` to write your findings into the `{self.root_path}/manifests/phase_1/manifest.json`)!
        
        Inspect the `{self.root_path}/src/` subdirectories and/or files using the `file_system_search_agent` or `data_file_inspector_agent`; some default empty files have been set up for you. You can use these if you need, potentially with the `file_managing_agent`.
        """

    def eda_preprocessing_directive_2(self) -> str:
        """
        Constructs the detailed agent directive string for EDA and preprocessing.
        Returns:
            A string containing the agent directive.
        """

        return f"""
        {self.initial_prompt}

        Your current objective: Phase 2 - EDA and Data Preprocessing.

        Follow these general steps, using your available sub-agents and reasoning capabilities. Finish this phase to the best
        of your ability! Keep in mind this is a high stakes project, everything you do will have impact further down the line.
        You are at liberty to use as many sub-agents as you want to perform the exploratory data analysis as professionally as possible.
        Use the `Browse_agent` if needed for research or to find EDA techniques.

        1.  **Plan and Execute EDA:**
            * Based on `Data Type` and inspection, plan EDA. If the plan is unclear or you need to make choices with significant impact (e.g., how to handle a large amount of missing data), use `ask_user_for_input`.
            * Generate Python EDA script. Use the `file_managing_agent` to save it to `{self.root_path}/scripts/exploratory_data_analysis.py`.
                * The script should print summaries and save plots. Use the `file_managing_agent` to create the directory `{self.root_path}/results/plots/eda/` if it doesn't exist, and ensure the script saves plots there.
                * Optionally, the script can save an `eda_report.json`. Use the `file_managing_agent` to save it to `{self.root_path}/data/interim/`.
            * If you identify the need for new Python libraries for the EDA script:
                * First, use the `data_file_inspector_agent` to check if `requirements.txt` exists and if the libraries are already listed.
                * If not listed or the file doesn't exist, use the `package_installing_agent` to add them to `{self.root_path}/requirements.txt`. Ensure each library is on a new line.
            * Execute EDA script using `execute_python_script` (this is an orchestrator capability).
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
            * Summarize all actions taken, key findings from EDA, the current state of the data, and the preprocessing plan in the scratchpad.txt.
            * Indicate that Phase 2 (EDA and Data Preprocessing) is complete and you are ready for the next phase.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors, or need a decision.
        Start by using the `data_file_inspector_agent` to read the scratchpad.txt for any existing notes and the manifest from the previous phase under `{self.root_path}/manifests/phase_1/manifest.json`.
        It is imperative that you end the phase by using the `file_managing_agent` to write your findings into `{self.root_path}/manifests/phase_2/manifest.json`)!
        """

    def data_preprocessing_and_feature_engineering_3(self) -> str:
        """
        Constructs the detailed agent directive string for implementing data preprocessing and feature engineering.
        """
        return f"""
        {self.initial_prompt}

        Your current objective: Phase 3 - Data Preprocessing Implementation and Feature Engineering.

        Follow these general steps, using your available sub-agents and reasoning capabilities.
        Start by using the `data_file_inspector_agent` to read {self.scratchpad_file}, and the manifest from Phase 2: `{self.root_path}/manifests/phase_2/manifest.json`.

        1.  **Implement Preprocessing Script:**
            * Add tasks: "Develop data preprocessing script", "Execute preprocessing script", "Verify processed data".
            * Based on the detailed plan in the Phase 2 manifest, write a Python script. Use the `file_managing_agent` to save it (e.g., as `{self.root_path}/scripts/data_preprocessing.py`).
                * This script should load raw or interim data (as identified in previous phases).
                * Apply all planned cleaning, transformation, encoding, and scaling steps.
                * Save the processed data to a designated output directory (e.g., `{self.root_path}/data/processed/`). Use the `file_managing_agent` to create this directory. Ensure output format is suitable for model training (e.g., CSV, Parquet, NumPy arrays).
            * Check and update `{self.root_path}/requirements.txt`. Use the `data_file_inspector_agent` to read it, and the `package_installing_agent` to append new libraries if needed.
            * Execute the script using `execute_python_script` (orchestrator capability). Analyze output/errors. Debug or use `ask_user_for_input` if issues arise.
            * Log script location, processed data path, and any important observations in {self.scratchpad_file}.

        2.  **Plan and Implement Feature Engineering (if applicable):**
            * Add task: "Plan and implement feature engineering".
            * Review EDA findings and the `Project Task`. Identify opportunities for feature engineering.
            * If not already part of the preprocessing script, you might create a separate script (e.g., `{self.root_path}/scripts/feature_engineering.py`) or modify the existing one. Use the `file_managing_agent` to manage this script file.
            * If the strategy for feature engineering is complex or has many alternatives, briefly outline your proposal in the scratchpad and use `ask_user_for_input` for confirmation or guidance.
            * Implement the chosen feature engineering steps. This might involve re-running parts of your preprocessing script or executing a new one.
            * Ensure engineered features are saved alongside or as part of the processed dataset.
            * Log the feature engineering plan, implementation details, and outcomes in {self.scratchpad_file}.

        3.  **Verify Processed Data Quality:**
            * Add task: "Verify quality and structure of processed data".
            * Perform a quick inspection of the processed data. Use the `data_file_inspector_agent` to load and check shape, data types, and a sample of the processed data.
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
            * Use the `file_managing_agent` to create this manifest.
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

        Follow these general steps using your available sub-agents.
        Start by using the `data_file_inspector_agent` to read {self.scratchpad_file} and the manifest from Phase 3: `{self.root_path}/manifests/phase_3/manifest.json`.

        1.  **Review Processed Data and Task for Model Selection:**
            * Use the `data_file_inspector_agent` to load information about the processed data from the Phase 3 manifest.
            * Based on this, research suitable machine learning models. Use the `Browse_agent` (utilizing its search tools like arXiv, Google Scholar, GitHub) for ideas or implementations if needed.
            * This is the most important step for a successful completion of this phase, make sure to use the scratchpad to write down your thoughts and research which model is the best one for this task.
            * Select a candidate model appropriate for the task. Justify your choices in {self.scratchpad_file}. If uncertain about choices, use `ask_user_for_input`.

        2.  **Develop Training Script(s):**
            * Create a Python script (e.g., `{self.root_path}/src/model_training.py`). Use the `file_managing_agent` to write this file.
            * The script should:
                * Load the processed data (path from Phase 3 manifest).
                * Implement data splitting (e.g., train, validation, test sets). Ensure this is done consistently.
                * Define and implement the selected model architectures using the `Target Framework`.
                * Set up the training loop, including loss function, optimizer, and relevant metrics.
                * Include functionality to save trained model artifacts and training logs/metrics. Use the `file_managing_agent` to create directories like `{self.root_path}/models/model_name/` and `{self.root_path}/results/training_logs/`.
            * Check and update `{self.root_path}/requirements.txt` for any new libraries (e.g., scikit-learn, torch). Use the `data_file_inspector_agent` to read the file and the `package_installing_agent` to append new libraries.
            * Use functions, typing and classes. Make sure everything is modularized.

        3.  **Perform Initial Training Runs:**
            * Add task: "Execute initial training for candidate models".
            * Execute the training script for each candidate model using `execute_python_script` (orchestrator capability).
            * Monitor the training process (if possible, through script output).
            * If training fails or produces unexpected errors, analyze the output, debug the script, or use `ask_user_for_input` for assistance.
            * Log paths to saved models, training logs, and key performance metrics (e.g., training/validation loss and accuracy) in {self.scratchpad_file}. You can use `log_agent_message` for detailed logs if useful (orchestrator capability).

        4.  **Conclude Phase 4:**
            * Summarize actions, selected models, training setup, and initial training results in {self.scratchpad_file}.
            * Create the manifest: `{self.root_path}/manifests/phase_4/manifest.json`. This JSON file should include:
                * List of models trained.
                * Path to the training script(s).
                * Details of data splits.
                * Paths to saved model artifacts for each model.
                * Key initial training/validation metrics for each model.
                * Path to any training log files.
            * Use the `file_managing_agent` to create this manifest.
            * Indicate in {self.scratchpad_file} that Phase 4 is complete.

        Always provide clear reasoning for your actions. Use the scratchpad.txt for detailed logging.
        Do not hesitate to use `ask_user_for_input` when you face ambiguity, critical errors, or need a decision.
        Use the `Browse_agent` for inspiration on which type of model or method should be best suited for this type of task; work to the best of your ability to improve the model.
        """