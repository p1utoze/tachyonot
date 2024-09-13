import os
from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).parent.parent / "resources"

def get_model_dir():
    try:
        return Path(os.environ["MODEL_DIR"])
    except KeyError:
        print("No MODEL_DIR env variable found!. Setting Default dir")
        return ROOT_DIR / "model"

def download_nltk_data():
    if not os.path.exists(ROOT_DIR / "nltk_data"):
        os.makedirs(Path.home() / "nltk_data", exist_ok=True)
        shutil.move(Path.home() / "nltk_data", ROOT_DIR / "nltk_data")

MODEL_DIR = get_model_dir()
download_nltk_data()

embedding_model = MODEL_DIR / "all-MiniLM-L6-v2.F32.gguf"
model_path = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
storage_path =  ROOT_DIR / "storage"
whipser_path = MODEL_DIR

chunk_size = 1024
embedding_dimension = 384

max_new_tokens = 512
top_k = 1
top_p =0.92
temperature = 0.5
n_ctx = 2048

log_level = "INFO"