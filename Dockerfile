# Use an official Python runtime as a parent image
FROM debian:b

# Set the working directory in the container
WORKDIR /app

#RUN apt-get update && apt-get install -y libegl1 libdbus-1-3 libxkbcommon-x11-0 \
#     libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
#     libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 \
#     x11-utils libxcb-cursor0 libopengl0 libegl1-mesa

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
    libegl1 \
#    libxcb-cursor-dev \
    python3-cffi-backend \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for musl
RUN ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1

# Copy the current directory contents into the container at /app
COPY model/ /app/model/
#COPY config.yaml /app/
COPY dist/tachyonot-3.5.1-py3-none-any.whl /app/tachyonot-3.5.1-py3-none-any.whl
COPY gui.py /app/
COPY .tiktoken_cache/ /app/.tiktoken_cache/

# Set environment variable
#ENV CONFIG_PATH=/app/config.yaml

# Create cache directory
RUN mkdir .cache

## Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

RUN . /app/.venv/bin/activate 
# Install Python dependencies
RUN TMPDIR=.cache CMAKE_ARGS="-DLLAMA_NATIVE=off -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off" pip install llama-cpp-python==0.2.88 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN TMPDIR=.cache pip install pip==24.0 wheel webrtcvad --cache-dir .cache
#RUN TMPDIR=.cache pip install numpy==1.26.0 --cache-dir .cache
RUN TMPDIR=.cache pip install  -C cmake.define.FAISS_OPT_LEVEL=generic -C cmake.define.DLLAMA_AVX2=off -C cmake.define.WHISPER_NO_AVX512=off -C cmake.define.WHISPER_NO_AVX512_VBMI=off -C cmake.define.WHISPER_NO_AVX512_VNNI=off tachyonot-3.5.1-py3-none-any.whl  --cache-dir .cache  --ignore-installed PyYAML --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN TMPDIR=.cache pip install nltk==3.8.1 --cache-dir .cache
RUN TMPDIR=.cache pip install pyinstaller --cache-dir .cache

RUN python3 --version

# Download NLTK data
RUN python3 -m nltk.downloader punkt wordnet stopwords punkt_tab

ENV TIKTOKEN_CACHE_DIR="$(readlink -f ./gui/_internal/.tiktoken_cache/)"

RUN pyinstaller gui.py -D --name "tachQt" --hidden-import=_cffi_backend --collect-binaries llama_cpp --hidden-import=tiktoken_ext.openai_public --hidden-import=tiktoken_ext --add-data=/app/.tiktoken_cache/:./.tiktoken_cache/
