from yaml import safe_load
from pathlib import Path
from typing import Optional, Union
repo_id = "epochalypse/hmi-models"
default_dtype = "int4"
DEFAULT_MODELS_YAML = Path(__file__).parent.parent / "models.yaml"


class ModelConfig:
    def __init__(self, file_path: Union[Path, str] = DEFAULT_MODELS_YAML):
        self.yaml_data = safe_load(open(file_path, "r"))
        self._model_config = None
        self.models_ = self.load_models_list()

    def load_models_list(self):
        try:
            models_list = self.yaml_data["models"]["text2text"].keys()
        except KeyError:
            models_list = ["Phi3Mini-4K", "qwen", "tinyLlama", "tinyMistral"]
        return models_list

    def load_model_config(self, model_name: str, dtype: Optional[str] = None):
        self._model_config = self.yaml_data["models"]["text2text"][model_name]

    def get_available_dtypes(self, model_name: str):
        return self._model_config["dtypes"]

    def validate_dtype(self, dtype: Optional[str] = None):
        if dtype is None:
            dtype = default_dtype

        if dtype not in self._model_config["dtypes"]:
            raise ValueError(f"Data type {dtype} not found in {self._model_config['dtypes']}")
        return dtype

    def get_base_model_tokenizer(self, model_name: str):
        try:
            return self._model_config["hfId"]
        except (AttributeError, KeyError):
            raise ValueError(f"Model {model_name} not found in {self.models_}")





ENV_ERROR_MESSAGE = ("Please set the HF_TOKEN environment variable by running \n`export HF_TOKEN=<your_token>`\nOr use "
                     "the -t option to pass the token")
