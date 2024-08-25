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
    && rm -rf /var/lib/apt/lists/*

# Create symlink for musl
RUN ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1

# Copy the current directory contents into the container at /app
COPY model/ /app/model/
COPY config.yaml /app/
COPY dist/simatic-3.1.2-py3-none-any.whl /app/simatic-3.1.2-py3-none-any.whl
COPY vm_panel_setup.sh app/
COPY data/ app/data/

# Set environment variable
ENV CONFIG_PATH=/app/config.yaml

# Create cache directory
RUN mkdir .cache

## Create and activate virtual environment
#RUN python3 -m venv .venv
#ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
RUN TMPDIR=.cache CMAKE_ARGS="-DLLAMA_NATIVE=off" pip install llama-cpp-python==0.2.78 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN TMPDIR=.cache pip3 install wheel webrtcvad --cache-dir .cache
RUN TMPDIR=.cache pip install numpy==1.26.0 --cache-dir .cache
RUN TMPDIR=.cache pip install simatic-3.1.2-py3-none-any.whl  --cache-dir .cache --extra-index-url https://download.pytorch.org/whl/cpu --ignore-installed PyYAML
RUN TMPDIR=.cache pip install scalene --cache-dir .cache
RUN TMPDIR=.cache pip install nltk==3.8.1 --cache-dir .cache

# Download NLTK data
RUN python3 -m nltk.downloader punkt wordnet stopwords punkt_tab

# Run app.py when the container launches
#CMD ["python", "app.py"]