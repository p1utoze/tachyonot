#!/bin/bash

# Set environment variables

export HF_HUB_ENABLE_HF_TRANSFER=1

readarray -t t2tmodels < <( yq '.models["text2text"] | keys | .[]' models.yaml)
models_string="${t2tmodels[*]}"

CLEAR='\033[0m'
RED='\033[0;31m'


join () {
  local IFS="$1"
  shift
  echo "$*"
}

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}👉 $1${CLEAR}\n";
  fi
  echo "Usage: $0 [-m model] [-hf token] \"<prompt>\""
  printf "\nFlag arguments:\n"
  echo "  -m, --model       {$(join , "${t2tmodels[@]}")} The exact model name from this model repository: https://huggingface.co/epochalypse/hmi-models"
  echo "  -hf, --hf-token       Use the Hugging Face API token for inference. (Requires HF_HUB_TOKEN environment variable to be set)"
  printf "\nPositional arguments:\n"
  echo "  prompt        The prompt to generate text from in string format."
  echo
  echo "Example: $0 -m Phi3Mini-4K -hf <your_token> \"Once upon a time, there was a \""
  echo "NOTE: You must have access to the Hugging Face model repository (Epochalypse/hmi-models) to use the HF token."
  exit 1
}

#Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python3."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    python3 -m pip install virtualenv
fi

# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    virtualenv venv
fi

# parse params
while [[ "$#" -gt 0 ]]; do case $1 in
  -m|--model) MODEL="$2"; shift; shift;;
  -hf|--hf-token) export HF_TOKEN=$2; shift; shift;;
  --dry-run) DRY_RUN=1; shift;;
  -h|--help) usage; shift;;
  *)
    if [ -z "$PROMPT" ]; then
      PROMPT="$1"; shift
    else
      usage "Unknown parameter passed: $1"; shift; shift
    fi;;
esac; done


if [[ "$DRY_RUN" -gt 0 ]]; then
    echo "Dry run completed."
    exit 0
fi

# Activate the virtual environment
source venv/bin/activate

# Check if requirements.txt file exists
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt file not found."
    deactivate
    exit 1
else
    # Install packages from requirements.txt
    while read -r package; do
        package_name=$(echo "$package" | cut -d'=' -f1 | sed 's/[><^]//g')
#        package_version=$(echo "$package" | cut -d'=' -f2)
        # Check if the package is installed with the correct version
        # shellcheck disable=SC1073
        if [[ "${package_name}" =~ ^-+ ]]; then
            continue

        elif ! pip freeze | grep -q "^$package_name" ; then
            echo "Installing $package_name"
            pip install "$package"
        fi
    done < requirements.txt
fi



# Validate argument types
if [ -z "$PROMPT" ] ; then
    usage "Error: Prompt is required."; echo
    deactivate
    exit 1
fi

if [ -z "$MODEL" ] ; then
    usage "Error: Model is required."; echo
    deactivate
    exit 1
fi



# shellcheck disable=SC2076
if [[ ! " ${t2tmodels[*]} " =~ " ${MODEL} " ]]; then
    echo "Error: Invalid model name --> ${MODEL}. Valid models are: $(join , ${t2tmodels[*]})"
    deactivate
    exit 1
fi

echo "Model: $MODEL"
echo "Prompt: $PROMPT"

# Run the Python inference function
#python3 inference.py --model "$MODEL" --hf-token "$HF_TOKEN" --prompt "$PROMPT"


# Deactivate the virtual environment
deactivate
