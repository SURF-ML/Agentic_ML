import os
import json
import argparse 
import yaml
import logging

from agentic_ml.utils.util_functions import load_config

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_project_structure(root_dir, project_folder): # root_dir is now directly the project name
    """
    Creates the skeletal folder and file structure for a machine learning project.
    """
    if os.path.exists(root_dir):
        print(f"Directory '{root_dir}' already exists. Please remove or choose a different name.")
        # Potentially add logic here for the agent to handle existing directories,
        # e.g., by creating a uniquely named directory or asking for confirmation to overwrite.
        # For now, we'll just exit to prevent accidental overwrites.
        return

    print(f"Creating project structure in '{root_dir}'...")

    # Define directories and files to create
    # Using None for files that should just be empty directories
    # Using an empty string "" for files that should be created empty
    # Using a string for files that should have minimal initial content
    structure = {project_folder: {".gitignore": "# Python\n__pycache__/\n*.py[cod]\n*$py.class\n\n# Environments\n.env\n.venv\nvenv/\nENV/\nenv/\n\n# IDEs and editors\n.idea/\n.vscode/\n*.swp\n\n# Data files - uncomment if you want to ignore common data types\n# *.csv\n# *.json\n# *.parquet\n# *.pkl\n\n# Large model files - uncomment if you store models locally\n# *.pt\n# *.pth\n# *.h5\n# *.onnx\n\n# Results and Logs - uncomment if you want to ignore\n# results/logs/*\n# results/trained_models/*\n\n# OS-specific\n.DS_Store\nThumbs.db",
            "README.md": f"# Project: {os.path.basename(root_dir)}\n\nThis project was initialized by an AI agent for a machine learning task.\n\n## Setup\n\n```bash\npip install -r requirements.txt\n```\n\n## Structure\n\n- `config/`: Configuration files.\n- `data/`: Raw, processed, and interim data.\n- `scripts/`: Utility scripts.\n- `src/`: Main source code.\n  - `agent/`: Agent-specific files (Do's and Don'ts, scratchpad).\n  - `data_preprocessing/`: Data loading and preprocessing modules.\n  - `datasets/`: Custom dataset classes.\n  - `models/`: Model architectures.\n  - `training/`: Training scripts and utilities.\n  - `evaluation/`: Evaluation metrics and scripts.\n  - `utils/`: General utility functions.\n  - `main.py`: Main executable script for the pipeline.\n- `tests/`: Unit and integration tests.\n- `results/`: Output directories for models, logs, and plots.\n- `docs/`: Project documentation.",
            "requirements.txt": "# Add project dependencies here\n# Example:\n# pandas\n# numpy\n# scikit-learn\n# torch\n# torchvision\n# tensorflow",
            "config": {
                "main_config.yaml": f"# Global project configurations\nproject_name: {os.path.basename(root_dir)}\n# random_seed: 42\n# paths:\n#   data_raw: data/raw/\n#   data_processed: data/processed/",

            },
            "data": {
                "raw": None,
                "processed": None,
                "interim": None,
                "external": None
            },
            "scripts": {
            },
            "src": {
                "__init__.py": "",
                "data_preprocessing": {
                    "__init__.py": "",
                },
                "datasets": {
                    "__init__.py": "",
                },
                "models": {
                    "__init__.py": "",
                },
                "training": {
                    "__init__.py": "",
                },
                "evaluation": {
                    "__init__.py": "",
                },
                "utils": {
                    "__init__.py": "",
                },
            },
            "tests": {
                "__init__.py": "",
            },
            "results": {
                "trained_models": None,
                "logs": None,
                "plots": None
            
            },
            "manifests": {
            },
            "agent": {
            },
            "docs": None
        },
        "raw_data": None
    }
    # Function to recursively create directories and files
    def create_items(current_path, items):
        for name, content in items.items():
            item_path = os.path.join(current_path, name)
            if isinstance(content, dict):
                os.makedirs(item_path, exist_ok=True)
                create_items(item_path, content)
                print(f"  Created directory: {item_path}/")
            elif content is None:
                os.makedirs(item_path, exist_ok=True)
                print(f"  Created directory: {item_path}/")
            else:
                with open(item_path, 'w') as f:
                    if content:
                        f.write(content)
                print(f"  Created file:      {item_path}")

    # Create the root directory first
    os.makedirs(root_dir, exist_ok=True)
    print(f"Created root directory: {root_dir}/")

    # Create all other items
    create_items(root_dir, structure)

    print("\nProject structure created successfully!")
    print(f"Next steps for the agent might involve populating 'requirements.txt' and then running 'pip install -r {os.path.join(root_dir, 'requirements.txt')}'.")
    print("The agent should then start implementing the functionalities outlined in the 'src/' directory, guided by the user's prompt.")


def main():
    parser = argparse.ArgumentParser(description="Create a skeletal structure for a new Machine Learning project.")
    parser.add_argument("--config", help="The name of the project root directory to be created.")
    parser.add_argument("--project_path", help="The name of the project root directory to be created.")

    args = parser.parse_args()

    # Use the project_name from the command-line arguments
    create_project_structure("../{args.project_name}", f"{args.project_name}")

if __name__ == "__main__":
    main()
