#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 --prompt-file <prompt_file_path> --config-file <config_file_path> [--debug]"
    exit 1
}

# Parse command line arguments
debug_mode=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt-file) prompt_file_path="$2"; shift ;;
        --config-file) config_file_path="$2"; shift ;;
        --debug) debug_mode=true ;;
        *) echo "Unknown parameter passed: $1"; print_usage ;;
    esac
    shift
done

# Enable debug mode if requested
if $debug_mode; then
    set -x
fi

# Check if both required arguments are provided
if [ -z "$prompt_file_path" ] || [ -z "$config_file_path" ]; then
    echo "Error: Both --prompt-file and --config-file must be provided."
    print_usage
fi

# Check if the prompt file exists and is readable
if [[ ! -f "$prompt_file_path" ]]; then
    echo "Error: Prompt file not found: $prompt_file_path"
    exit 1
elif [[ ! -r "$prompt_file_path" ]]; then
    echo "Error: Cannot read prompt file: $prompt_file_path"
    exit 1
fi

# Check if the prompt file is empty
if [[ ! -s "$prompt_file_path" ]]; then
    echo "Error: Prompt file is empty: $prompt_file_path"
    exit 1
fi

if $debug_mode; then
    echo "Contents of prompt file:"
    cat "$prompt_file_path"
fi

# Initialize prompt counter
prompt_counter=1

# Read the file line by line
while IFS= read -r line || [[ -n "$line" ]]; do
    echo "Processing prompt $prompt_counter"

    mkdir -p "stress_results/prompt $prompt_counter"

    # Define the output file path
    output_file="stress_results/prompt '$prompt_counter'/profile.html"

    # Run the simatic CLI command
    simatic_command="python3 -m scalene --profile-all --no-browser --json --outfile '$output_file' -m --- simatic chat '$line' --config-path '$config_file_path'"
    if $debug_mode; then
        echo "Executing command: $simatic_command"
    fi
    eval $simatic_command

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Prompt $prompt_counter processed successfully"
    else
        echo "Error processing prompt $prompt_counter"
        echo "Command exit status: $?"
    fi

    ((prompt_counter++))
done < "$prompt_file_path"

echo "All prompts processed."

# Disable debug mode if it was enabled
if $debug_mode; then
    set +x
fi