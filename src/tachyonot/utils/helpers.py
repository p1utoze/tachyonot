import os
import pywhispercpp.constants as constants
from chardet import detect
from tachyonot.utils.config import ROOT_DIR
from rich import print
import PyPDF2
import pandas as pd
from docx import Document


def set_tiktoken_env():
    """
    Set environment variables from argparse
    """
    tiktoken_dir = ROOT_DIR / ".tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_dir)
    print(os.environ["TIKTOKEN_CACHE_DIR"])


def _get_params(args) -> dict:
    """
    Helper function to get params from argparse as a `dict`
    """
    params = {}
    for arg in args.__dict__:
        if arg in constants.PARAMS_SCHEMA.keys() and getattr(args, arg) is not None:
            if constants.PARAMS_SCHEMA[arg]["type"] is bool:
                if getattr(args, arg).lower() == "false":
                    params[arg] = False
                else:
                    params[arg] = True
            else:
                params[arg] = constants.PARAMS_SCHEMA[arg]["type"](getattr(args, arg))
    return params


def get_config_path():
    try:
        config_path = os.environ["CONFIG_PATH"]
    except KeyError:
        raise ValueError("CONFIG_PATH environment variable not set")
    return config_path

def read_text_file(file_path):
    with open(file_path, "rb") as file:
        text = file.read().decode(detect(file.read())["encoding"])
    return text

def read_pdf_file(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def read_docx_file(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_xlsx_file(file_path):
    df = pd.read_excel(file_path)
    threshold = int(len(df) * 0.9)
    df = df.dropna(axis=1, thresh=threshold)
    df = df.dropna()
    return df

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".txt":
        text = read_text_file(file_path)
    elif file_extension == ".pdf":
        text = read_pdf_file(file_path)
    elif file_extension == ".docx":
        text = read_docx_file(file_path)
    elif file_extension == ".xlsx":
        text = read_xlsx_file(file_path)
    else:
        raise ValueError("Unsupported file type")

    return text