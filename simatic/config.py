from yaml import safe_load
from pathlib import Path
from typing import Optional, Union
from functools import cached_property, cache
from sys import platform
from os import getenv

home_dir = None
if platform == "win32":
    home_dir = getenv("USERPROFILE")
elif platform == "linux":
    home_dir = getenv("HOME")

CACHE_DIR = Path(home_dir) / ".cache" / "simatic"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

repo_id = "epochalypse/hmi-models"
default_dtype = "int4"
DEFAULT_MODELS_YAML = Path(__file__).parent.parent / "models.yaml"

SYSTEM_PROMPT = """
You are a Friendly help-desk assistant. You must handle any type of customer questions diligently. If unambigious, ask for more information. If you are unable to answer, escalate to a human agent.
"""


class ModelConfig:
    def __init__(self, file_path: Union[Path, str] = DEFAULT_MODELS_YAML):
        self.yaml_data = safe_load(open(file_path, "r"))
        self._model_config = None
        self.models_ = self.load_models_list

    @cached_property
    def load_models_list(self):
        try:
            models_list = self.yaml_data["models"]["text2text"].keys()
        except KeyError:
            models_list = ["Phi3Mini-4K", "qwen", "tinyLlama", "tinyMistral"]
        return models_list

    def load_model_config(self, model_name: str, dtype: Optional[str] = None):
        self._model_config = self.yaml_data["models"]["text2text"][model_name]

    @cached_property
    def get_available_dtypes(self):
        return self._model_config["dtypes"]

    @cache
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


if __name__ == "__main__":
    model_config = ModelConfig()
    print(model_config.load_models_list)
