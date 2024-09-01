# Use an official Python runtime as a parent image
FROM debian:bullseye-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    musl-dev \
    libportaudio2 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
   patchelf \
   ccache \
    clang \
    python3-cffi-backend \ 
    && rm -rf /var/lib/apt/lists/*

# Create symlink for musl
RUN ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1

# Copy the current directory contents into the container at /app
COPY model/ /app/model/
COPY config.yaml /app/
COPY dist/tachyonot-3.5.0-py3-none-any.whl /app/tachyonot-3.5.0-py3-none-any.whl
COPY cli.py /app/
COPY .tiktoken_cache/ /app/.tiktoken_cache/

# Set environment variable
ENV CONFIG_PATH=/app/config.yaml

# Create cache directory
RUN mkdir .cache

## Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

RUN . /app/.venv/bin/activate 
# Install Python dependencies
RUN TMPDIR=.cache CMAKE_ARGS="-DLLAMA_NATIVE=on" pip install llama-cpp-python==0.2.89 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN TMPDIR=.cache pip3 install wheel webrtcvad --cache-dir .cache
#RUN TMPDIR=.cache pip install numpy==1.26.0 --cache-dir .cache
RUN TMPDIR=.cache pip install tachyonot-3.5.0-py3-none-any.whl  --cache-dir .cache  --ignore-installed PyYAML
RUN TMPDIR=.cache pip install nltk==3.8.1 --cache-dir .cache
RUN TMPDIR=.cache pip install pyinstaller --cache-dir .cache

RUN python3 --version

# Download NLTK data
RUN python3 -m nltk.downloader punkt wordnet stopwords punkt_tab

RUN pyinstaller cli.py -F --hidden-import=_cffi_backend --collect-binaries llama_cpp --hidden-import=tiktoken_ext.openai_public --hidden-import=tiktoken_ext --add-data=/app/.tiktoken_cache/:./.tiktoken_cache/
