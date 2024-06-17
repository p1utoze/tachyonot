import os
from click import prompt


def get_hf_token():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        hf_token = prompt("Please enter your Hugging Face API token", hide_input=True)
    return hf_token


def get_prompt_template(text_input: str):
    messages = [
        {
            "role": "system",
            "content": "You are a Captain of the ship, who always responds in the style of a pirate accent",
        },
        {"role": "user", "content": text_input},
    ]
    return messages