import os
import json
import argparse 
import yaml
import logging

from utils.util_functions import load_config

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
                "model_config.yaml": "# Model-specific hyperparameters\n# model_name: MyModel\n# learning_rate: 0.001\n# batch_size: 32\n# epochs: 100",
                "data_config.yaml": "# Data-specific configurations\n# target_variable: target\n# features_to_drop: []\n# test_split_ratio: 0.2"
            },
            "data": {
                "raw": None,
                "processed": None,
                "interim": None,
                "external": None
            },
            "scripts": {
                "download_data.sh": "#!/bin/bash\n# Script to download data\n# echo 'Downloading data...'\n# Example: wget -P data/raw/ http://example.com/data.zip",
                "preprocess_data.py": "def main():\n    print(\"Preprocessing data...\")\n    # Add data preprocessing logic here\n\nif __name__ == \"__main__\":\n    main()",
                "run_evaluation.py": "def main():\n    print(\"Running evaluation...\")\n    # Add evaluation logic here\n\nif __name__ == \"__main__\":\n    main()"
            },
            "src": {
                "__init__.py": "",
                "data_preprocessing": {
                    "__init__.py": "",
                    "loader.py": "class DataLoader:\n    def __init__(self, data_path):\n        self.data_path = data_path\n\n    def load_data(self):\n        # TODO: Implement data loading logic based on data_path\n        raise NotImplementedError(\"load_data method not implemented\")\n\n# Example usage:\n# if __name__ == '__main__':\n#     loader = DataLoader('path/to/your/data')\n#     data = loader.load_data()",
                    "augmentations.py": "# Functions for data augmentation\ndef example_augmentation(data):\n    # TODO: Implement an example data augmentation technique\n    raise NotImplementedError(\"example_augmentation function not implemented\")",
                    "preprocessing.py": "def preprocess_features(data):\n    # TODO: Implement feature preprocessing logic\n    raise NotImplementedError(\"preprocess_features function not implemented\")\n\ndef clean_data(data):\n    # TODO: Implement data cleaning logic (e.g., handling missing values)\n    raise NotImplementedError(\"clean_data function not implemented\")"
                },
                "datasets": {
                    "__init__.py": "",
                    "custom_dataset.py": "# Example for PyTorch:\n# from torch.utils.data import Dataset\n# class CustomDataset(Dataset):\n#     def __init__(self, data, targets, transform=None):\n#         self.data = data\n#         self.targets = targets\n#         self.transform = transform\n# \n#     def __len__(self):\n#         return len(self.data)\n# \n#     def __getitem__(self, idx):\n#         sample = self.data[idx]\n#         target = self.targets[idx]\n#         if self.transform:\n#             sample = self.transform(sample)\n#         return sample, target\n\n# Example for TensorFlow:\n# import tensorflow as tf\n# def create_tf_dataset(features, labels, batch_size):\n#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))\n#     dataset = dataset.shuffle(buffer_size=len(features))\n#     dataset = dataset.batch(batch_size)\n#     dataset = dataset.prefetch(tf.data.AUTOTUNE)\n#     return dataset\n\nraise NotImplementedError(\"Skeletal dataset file. Implement based on ML framework.\")"
                },
                "models": {
                    "__init__.py": "",
                    "model_architecture.py": "# Example for PyTorch:\n# import torch\n# import torch.nn as nn\n# class MyModel(nn.Module):\n#     def __init__(self, input_dim, output_dim):\n#         super(MyModel, self).__init__()\n#         # TODO: Define model layers\n#         self.fc1 = nn.Linear(input_dim, 128)\n#         self.relu = nn.ReLU()\n#         self.fc2 = nn.Linear(128, output_dim)\n# \n#     def forward(self, x):\n#         # TODO: Define forward pass\n#         x = self.fc1(x)\n#         x = self.relu(x)\n#         x = self.fc2(x)\n#         return x\n\n# Example for TensorFlow/Keras:\n# import tensorflow as tf\n# def create_keras_model(input_shape, num_classes):\n#     model = tf.keras.Sequential([\n#         tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),\n#         # TODO: Add more layers\n#         tf.keras.layers.Dense(num_classes, activation='softmax')\n#     ])\n#     return model\n\nraise NotImplementedError(\"Skeletal model architecture file. Implement based on ML framework.\")"
                },
                "training": {
                    "__init__.py": "",
                    "trainer.py": "class Trainer:\n    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, epochs):\n        self.model = model\n        self.train_loader = train_loader\n        self.val_loader = val_loader\n        self.optimizer = optimizer\n        self.criterion = criterion\n        self.device = device\n        self.epochs = epochs\n\n    def train_epoch(self):\n        self.model.train()\n        # TODO: Implement one epoch of training\n        raise NotImplementedError(\"train_epoch method not implemented\")\n\n    def validate_epoch(self):\n        self.model.eval()\n        # TODO: Implement one epoch of validation\n        raise NotImplementedError(\"validate_epoch method not implemented\")\n\n    def fit(self):\n        for epoch in range(self.epochs):\n            print(f\"Epoch {epoch+1}/{self.epochs}\")\n            # self.train_epoch()\n            # self.validate_epoch()\n            # Add logging, saving checkpoints, etc.\n        raise NotImplementedError(\"fit method not fully implemented\")",
                    "losses.py": "# Custom loss functions can be defined here\n# Example:\n# import torch.nn as nn\n# class MyCustomLoss(nn.Module):\n#     def __init__(self):\n#         super(MyCustomLoss, self).__init__()\n# \n#     def forward(self, output, target):\n#         loss = ... # calculate loss\n#         return loss\npass",
                    "optimizers.py": "# Custom optimizer configurations or factory functions\n# Example:\n# import torch.optim as optim\n# def get_optimizer(model_parameters, optimizer_name='adam', lr=0.001):\n#     if optimizer_name.lower() == 'adam':\n#         return optim.Adam(model_parameters, lr=lr)\n#     elif optimizer_name.lower() == 'sgd':\n#         return optim.SGD(model_parameters, lr=lr, momentum=0.9)\n#     else:\n#         raise ValueError(f\"Optimizer {optimizer_name} not supported\")\npass"
                },
                "evaluation": {
                    "__init__.py": "",
                    "metrics.py": "# Define custom evaluation metrics\n# Example:\n# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n# def calculate_classification_metrics(y_true, y_pred):\n#     accuracy = accuracy_score(y_true, y_pred)\n#     precision = precision_score(y_true, y_pred, average='weighted') # Adjust average as needed\n#     recall = recall_score(y_true, y_pred, average='weighted')\n#     f1 = f1_score(y_true, y_pred, average='weighted')\n#     return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}\npass",
                    "evaluate_model.py": "def evaluate(model, test_loader, criterion, device):\n    model.eval()\n    total_loss = 0\n    all_preds = []\n    all_targets = []\n    # with torch.no_grad(): # if using PyTorch\n    #     for data, target in test_loader:\n    #         data, target = data.to(device), target.to(device)\n    #         output = model(data)\n    #         loss = criterion(output, target)\n    #         total_loss += loss.item()\n    #         # Store predictions and targets for metric calculation\n    #         all_preds.extend(output.argmax(dim=1).cpu().numpy())\n    #         all_targets.extend(target.cpu().numpy())\n    # avg_loss = total_loss / len(test_loader)\n    # metrics = calculate_classification_metrics(all_targets, all_preds) # from metrics.py\n    # print(f'Test Loss: {avg_loss}, Metrics: {metrics}')\n    # return avg_loss, metrics\n    raise NotImplementedError(\"evaluate function not implemented\")"
                },
                "utils": {
                    "__init__.py": "",
                    "logging_utils.py": "import logging\nimport sys\n\ndef setup_logger(name='project_logger', level=logging.INFO):\n    logger = logging.getLogger(name)\n    logger.setLevel(level)\n    handler = logging.StreamHandler(sys.stdout)\n    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')\n    handler.setFormatter(formatter)\n    if not logger.hasHandlers():\n        logger.addHandler(handler)\n    return logger\n\n# logger = setup_logger()",
                    "file_io_utils.py": "import json\nimport yaml\nimport pickle\n\ndef save_json(data, filepath):\n    with open(filepath, 'w') as f:\n        json.dump(data, f, indent=4)\n\ndef load_json(filepath):\n    with open(filepath, 'r') as f:\n        return json.load(f)\n\ndef save_yaml(data, filepath):\n    with open(filepath, 'w') as f:\n        yaml.dump(data, f)\n\ndef load_yaml(filepath):\n    with open(filepath, 'r') as f:\n        return yaml.safe_load(f)\n\ndef save_pickle(data, filepath):\n    with open(filepath, 'wb') as f:\n        pickle.dump(data, f)\n\ndef load_pickle(filepath):\n    with open(filepath, 'rb') as f:\n        return pickle.load(f)"
                },
                "main.py": "# Main script to run the ML pipeline\n# from config import main_config, model_config, data_config # Assuming you'll load these\n# from src.data_preprocessing.loader import DataLoader\n# from src.data_preprocessing.preprocessing import preprocess_features\n# from src.datasets.custom_dataset import CustomDataset # or your tf.data pipeline\n# from src.models.model_architecture import MyModel # or create_keras_model\n# from src.training.trainer import Trainer\n# from src.evaluation.evaluate_model import evaluate\n# from src.utils.logging_utils import setup_logger\n\n# logger = setup_logger()\n\ndef run_pipeline():\n    # logger.info(\"Starting ML Pipeline...\")\n\n    # 1. Load Configuration (handled by agent or defined here)\n    # print(\"Loading configurations...\")\n\n    # 2. Load Data\n    # print(\"Loading data...\")\n    # data_loader = DataLoader(main_config.paths.data_raw)\n    # raw_data = data_loader.load_data()\n\n    # 3. Preprocess Data\n    # print(\"Preprocessing data...\")\n    # processed_data = preprocess_features(raw_data)\n\n    # 4. Create Datasets/DataLoaders\n    # print(\"Creating datasets...\")\n    # train_dataset = CustomDataset(...)\n    # val_dataset = CustomDataset(...)\n    # test_dataset = CustomDataset(...)\n    # train_loader = ...\n    # val_loader = ...\n    # test_loader = ...\n\n    # 5. Initialize Model\n    # print(\"Initializing model...\")\n    # device = 'cuda' if torch.cuda.is_available() else 'cpu'\n    # model = MyModel(input_dim=..., output_dim=...).to(device)\n\n    # 6. Initialize Optimizer and Loss Function\n    # print(\"Setting up optimizer and loss...\")\n    # optimizer = ...\n    # criterion = ...\n\n    # 7. Train Model\n    # print(\"Starting training...\")\n    # trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, model_config.epochs)\n    # trainer.fit()\n\n    # 8. Evaluate Model\n    # print(\"Evaluating model...\")\n    # evaluate(model, test_loader, criterion, device)\n\n    # 9. Save Model and Results\n    # print(\"Saving results...\")\n\n    # logger.info(\"ML Pipeline finished.\")\n    raise NotImplementedError(\"run_pipeline function not implemented. Agent needs to fill this.\")\n\nif __name__ == '__main__':\n    run_pipeline()\n"
            },
            "tests": {
                "__init__.py": "",
                "test_data_preprocessing.py": "import unittest\n# from src.data_preprocessing import loader, preprocessing\n\nclass TestDataPreprocessing(unittest.TestCase):\n    def test_load_data(self):\n        # TODO: Add test for data loading\n        # loader_instance = loader.DataLoader('dummy_path')\n        # with self.assertRaises(NotImplementedError):\n        #     loader_instance.load_data()\n        pass\n\n    def test_preprocess_features(self):\n        # TODO: Add test for feature preprocessing\n        # with self.assertRaises(NotImplementedError):\n        #     preprocessing.preprocess_features(None)\n        pass\n\nif __name__ == '__main__':\n    unittest.main()",
                "test_datasets.py": "import unittest\n# from src.datasets import custom_dataset\n\nclass TestDatasets(unittest.TestCase):\n    def test_custom_dataset(self):\n        # TODO: Add test for custom dataset\n        # For example, check __len__ and __getitem__ if implemented\n        # with self.assertRaises(NotImplementedError):\n        #     # Attempt to instantiate or use the skeletal dataset parts\n        #     pass \n        pass\n\nif __name__ == '__main__':\n    unittest.main()",
                "test_models.py": "import unittest\n# from src.models import model_architecture\n\nclass TestModels(unittest.TestCase):\n    def test_model_creation(self):\n        # TODO: Add test for model instantiation and forward pass (checking shapes)\n        # with self.assertRaises(NotImplementedError):\n        #     # Attempt to instantiate or use the skeletal model parts\n        #     pass\n        pass\n\nif __name__ == '__main__':\n    unittest.main()",
                "test_training.py": "import unittest\n# from src.training import trainer\n\nclass TestTraining(unittest.TestCase):\n    def test_trainer_fit(self):\n        # TODO: Add test for trainer's fit method (mock objects for model, data, etc.)\n        # with self.assertRaises(NotImplementedError):\n        #     # Attempt to instantiate or use the skeletal trainer parts\n        #     mock_trainer = trainer.Trainer(None,None,None,None,None,None,None)\n        #     mock_trainer.fit()\n        pass\n\nif __name__ == '__main__':\n    unittest.main()"
            },
            "results": {
                "trained_models": None,
                "logs": None,
                "plots": None
            
            },
            "manifests": {
                "phase_0": None,
                "phase_1": None,
                "phase_2": None,
                "phase_3": None,
                "phase_4": None,
            },
            "agent": {
                "scratchpad.txt": "Init Agent Scratch Pad.",
                "agent_tasks.json": "[]"
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

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception:
        logger.critical(f"Failed to load configuration from {args.config}. Exiting.")
        return

    with open(config.get('run').get('initial_prompt_json'), "r") as f:
        project_json = json.load(f)

    # Use the project_name from the command-line arguments
    create_project_structure(project_json["project_path"], project_json["project_name"])

if __name__ == "__main__":
    main()
