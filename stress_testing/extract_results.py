import pandas as pd
import os
import json
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def remove_ansi_escape_sequences(text):
    """
    Remove ANSI escape sequences from text
    :param text: input text
    :return: text with ANSI escape sequences removed
    """
    ansi_escape_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape_pattern.sub("", text)


def extract_question_and_response(text):
    """
    Extract question and response from text
    :param text: input text
    :return: question and response
    """
    parts = text.split("Response:", 1)
    if len(parts) == 2:
        question = parts[0].strip().split("Question:", 1)
        response = parts[1].strip()
        return question[1], response
    else:
        return None, None


def load_profile_data(root_directory):
    """
    Load profile data from the root directory
    :param root_directory: root directory
    :return: list of profiles
    """
    profile_folders = [
        os.path.join(root_directory, x) for x in os.listdir(root_directory)
    ]
    profiles = []

    for folder in profile_folders:
        profile_info = {}
        profile_json_path = os.path.join(folder, "profile.json")
        output_txt_path = os.path.join(folder, "output.txt")

        if os.path.exists(profile_json_path):
            with open(profile_json_path, "r") as f:
                profile_info = json.load(f)

            if os.path.exists(output_txt_path):
                with open(output_txt_path, "r") as f:
                    cleaned_text = remove_ansi_escape_sequences(f.read())
                    question, response = extract_question_and_response(cleaned_text)
                    profile_info["prompt"] = question
                    profile_info["response"] = response

            profiles.append(profile_info)

    return profiles


def calculate_cpu_utilization(line_data):
    """
    Calculate CPU core utilization
    :param line_data: line data
    :return: CPU core utilization
    """
    return max(line["n_core_utilization"] for line in line_data)


def extract_metrics(profile):
    """
    Extract metrics from profile
    :param profile: profile data
    :return: extracted metrics
    """
    if "max_footprint_mb" not in profile:
        logging.warning(f"Missing 'max_footprint_mb' in profile: {profile}")
        return None

    llama_py_file = "/media/root-rw/venv/lib/python3.9/site-packages/llama_cpp/llama.py"

    return {
        "max_memory_consumption": profile["max_footprint_mb"],
        "growth_rate": profile["growth_rate"],
        "latency": profile["elapsed_time_sec"],
        "latency_by_model_initialization": profile["files"][llama_py_file][
            "percent_cpu_time"
        ],
        "cpu_utilization": calculate_cpu_utilization(
            profile["files"][llama_py_file]["lines"]
        ),
        "prompt": profile["prompt"],
        "response": profile["response"],
        "character_length": len(profile["response"]) if profile["response"] else 0,
    }


def main():
    root_directory = "profile_results"
    logging.info(f"Starting profile analysis from directory: {root_directory}")

    profile_data = load_profile_data(root_directory)
    logging.info(f"Loaded {len(profile_data)} profiles")

    results = []
    for i, profile in enumerate(profile_data, 1):
        metrics = extract_metrics(profile)
        if metrics:
            results.append(metrics)

        if i % 10 == 0 or i == 1:
            logging.info(f"Processed {i} prompts")

    logging.info(f"Analysis complete. Processed {len(results)} valid profiles")

    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            "max_memory_consumption": "Max Memory Consumption (MB)",
            "growth_rate": "Growth Rate (%)",
            "latency": "Latency (s)",
            "latency_by_model_initialization": "Latency due to model initialisation (s)",
            "cpu_utilization": "CPU Core Utilisation (%)",
            "prompt": "Prompt",
            "response": "Response",
            "character_length": "Character Length",
        }
    )
    print(df.head())

    output_file = "Simatic Profiling Results.xlsx"
    df.to_excel(output_file, index=False)
    logging.info(f"Saved profile data to {output_file}")


if __name__ == "__main__":
    main()
