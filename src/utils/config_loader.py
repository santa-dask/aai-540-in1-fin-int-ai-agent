# src/utils/config_loader.py
import yaml
import os
from functools import reduce
import operator
from dotenv import load_dotenv

# Constants for easy access to common config keys
PROJECT_ID = 'project.id'
REGION = 'project.region'
BQ_DATASET = 'bigquery.dataset'
RAW_BUCKET = 'data.raw_bucket'
CLEAN_BUCKET = 'data.clean_bucket'
WEIGHTS_BUCKET = 'data.weights_bucket'
RAW_BUCKET='data.raw_bucket'
CLEAN_BUCKET='data.clean_bucket'
WEIGHTS_BUCKET='data.weights_bucket'
RAW_FILE_PATH='data.raw_file_path'
TRAINING_FILE_PATH='data.training_file_path'
TEST_FILE_PATH='data.testing_file_path'
SAMPLED_FILE_PATH='data.sample_file_path'
MODEL_ID = "model.base_id"
HF_TOKEN = "model.hf_token"



class ConfigLoader:
    """
    A class to load, cache, and access configuration from a YAML file.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path=None):
        if not hasattr(self, 'config'):
            self.config = None
            self.config_path = config_path or self._get_default_config_path()
            self.load_config()

    def _get_default_config_path(self):
        """Calculates the default config path relative to the project root."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        return os.path.join(project_root, "config", "config.yaml")

    def load_config(self):
        """
        Loads the YAML configuration file and caches it.
        It also overrides the project ID from an environment variable if it exists.
        """
        load_dotenv()

        if self.config is not None:
            return

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Optional: Override GCP Project ID from Environment Variable
        gcp_project_env = os.getenv("GOOGLE_CLOUD_PROJECT")
        if gcp_project_env:
            if 'project' not in self.config:
                self.config['project'] = {}
            self.config['project']['id'] = gcp_project_env
            
    def get(self, key: str, default=None):
        """
        Retrieves a value from the loaded configuration using a dot-separated key.
        
        Args:
            key (str): The dot-separated key (e.g., "project.id").
            default: The value to return if the key is not found. Defaults to None.
            
        Returns:
            The configuration value or the default.
        """
        try:
            # Use reduce to navigate through the nested dictionary
            return reduce(operator.getitem, key.split('.'), self.config)
        except (KeyError, TypeError):
            return default

# Singleton instance for easy access
config_loader = ConfigLoader()

# Usage Example:
# When this script is run directly, it will load the default config 
# and print a value using the new get method.
if __name__ == "__main__":
    # Get a nested property
    lora_rank = config_loader.get("training.lora.r")
    print(f"LoRA Rank: {lora_rank}")

    # Get a top-level property
    project_id = config_loader.get("project.id")
    print(f"Project ID: {project_id}")