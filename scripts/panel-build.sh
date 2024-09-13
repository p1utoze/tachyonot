#!/bin/bash

printf "Before running the script make sure you have these files in /media/root-rw of the panel image\n
- pyproject.toml\n
- tachyonot package folder\n
- window.py (for testing)\n"

# Increase temp size
mount -o remount,size=1G /tmp

# Install essential packages
apt-get update && apt-get install -y     build-essential     cmake   libportaudio2 ffmpeg  musl-dev     python3-dev     python3-pip     python3-venv   libegl1 libgl-dev

curl -sSL https://install.python-poetry.org | python3 -

echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc

. ~/.bashrc

poetry env use 3.9
poetry config virtualenvs.path ~/.poetry/

cd /media/root-rw

source ~/.cache/pypoetry/virtualenvs/tachyonot-YW5wXK-Y-py3.9/bin/activate

CMAKE_ARGS="-DLLAMA_NATIVE=off -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off" pip install llama-cpp-python==0.2.78 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install  -C cmake.define.FAISS_OPT_LEVEL=generic --force-reinstall faiss-cpu
pip install   -C cmake.define.WHISPER_NO_AVX2=on -C cmake.define.WHISPER_NO_AVX=on -C cmake.define.WHISPER_NO_AVX512=on  -C cmake.define.WHISPER_NO_AVX512_VBMI=on -C cmake.define.WHISPER_NO_AVX512_VNNI=on pywhispercpp  --ignore-installed PyYAML --force-reinstall
pip install numpy==1.26.4 nltk==3.8.1

poetry install --without dev

echo "DEPENDENCIES SETUP DONE!"

poetry build --no-cache -o vm-dist

echo 'Build complete! Now copy the dist to your local!'


