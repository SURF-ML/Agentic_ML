def define_eda_preprocessing_directive(initial_prompt_details: dict) -> str:
    """
    Constructs the detailed agent directive string for EDA and preprocessing.
    Args:
        initial_prompt_details: Dictionary containing the initial user prompt details.
        scratchpad_file: Path to the scratchpad file the agent will use.
    Returns:
        A string containing the agent directive.
    """
    scratchpad_file = initial_prompt_details.get('scratchpad_path', './agent_environment/agent/files/scratchpad.txt')
    root_path = initial_prompt_details.get('project_path', 'N/A')

    return f"""
    Initial User Prompt Summary:
    Project: {initial_prompt_details.get('project_name', 'N/A')}
    Project root path: {initial_prompt_details.get('project_path', 'N/A')}
    Task: {initial_prompt_details.get('task_description', 'N/A')}
    Data Type: {initial_prompt_details.get('data_type', 'N/A')}
    Raw Data Location: '{initial_prompt_details.get('data_folder_path_from_user', 'N/A')}'
    Target Framework: {initial_prompt_details.get('target_framework', 'N/A')}
    Target Input Tensor Shape: {initial_prompt_details.get('target_tensor_shapes_input', 'N/A')}
    Scratchpad path: {scratchpad_file}

    You can also use the tools `read_scratchpad` and `update_scratchpad` for reading and writing in the scratchpad respectively.

    Assume at all times that you are working under {initial_prompt_details.get('project_path', 'N/A')},
    For instance, data/interim/ is found under {initial_prompt_details.get('project_path', 'N/A')}/data/interim and
    src/ is found under {initial_prompt_details.get('project_path', 'N/A')}/src, if you are unsure you can use `list_directory_contents`
    to inspect how your working directory looks like.

    Your current objective: Phase 1 - EDA and Data Preprocessing.

    Follow these general steps, using your available tools and reasoning capabilities.
    Use the Scratchpad to read detailed initial instructions and to log your progress, findings, and plans.

    You can use the scratchpad tools or you can use `log_agent_message` and `list_agent_log_files` for further 
    micro logging things you think are useful to save. Give the files appropriate names.

    1.  **Understand Task & Locate Data:**
        * The Raw Data Location contains the data you'll work with.
        * Confirm this path. If unusable/missing, then stop.
        * Log any understanding and stuff worth remembering using `update_scratchpad` tool.

    2.  **Initial Data Inspection:**
        * Use `list_directory_contents` on the Project root path (recursive, max_depth 2-3).
        * Identify a few sample files. For each, use `inspect_file_type_and_structure`.
        * Summarize and log findings.

    3.  **Plan and Execute EDA:**
        * Based on `Data Type` and inspection, plan EDA 
        * Generate Python EDA script, save to `{root_path}/scripts/exploratory_data_analysis.py` using `write_file`.
            * Print summaries, save plots to `{root_path}/results/plots/eda/` (`create_directory` if needed).
            * Optionally, save `eda_report.json` to `{root_path}/data/interim/`.
        * If you need to use new libraries (e.g., Pillow, pandas, matplotlib), append to `requirements.txt` (`append_to_file` after `read_file_content` to check).
        * Execute EDA script (`execute_python_script`). 

    4.  **Plan Preprocessing:**
        * Based on EDA and target tensor shape, plan preprocessing steps.
        * Plan manifest file (e.g., `{root_path}/data/interim/dataset_manifest.json`) if useful to create a Dataloader. The user specified the following
        about his choice of ML framework: {initial_prompt_details.get('target_framework', 'the specified ML framework')} .

    5.  **Conclude Phase 1:**
        * Summarize actions and data state in Scratchpad. Indicate readiness for next phase.

    Always provide clear reasoning and use the scratchpad for detailed logging. Start by reading the scratchpad.
    """

# 5.  **Generate Preprocessing Script(s):**
#         * Generate Python preprocessing code (update `/src/data_preprocessing/preprocessing.py` or new `src/data_preprocessing/build_processed_dataset.py`).
#         * Scripts process data from `current_data_path`, save to `/data/processed/`.
#         * Generate `dataset_manifest.json` if planned.
#         * Save script(s) using `write_file`.

#     6.  **Generate Tests for Preprocessing:**
#         * Generate unit tests for preprocessing, save to `tests/test_data_preprocessing.py` (`write_file`). Use fixtures in `tests/fixtures/` if needed (create with `write_file`).

#     7.  **Execute Preprocessing & Tests:**
#         * Run main preprocessing script (`execute_python_script`).
#         * Run unit tests (e.g., `execute_shell_command` with `python -m unittest tests/test_data_preprocessing.py`).
#         * Verify outputs in `data/processed/` and `data/interim/` (`list_directory_contents`). Log findings.