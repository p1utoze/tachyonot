sudo apt update
sudo apt install build-essential cmake python3-dev python3-pip -y
sudo apt install musl-dev -y
sudo ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1
sudo apt install python3-venv -y
cd /media/root-rw/
sudo apt-get install libportaudi  o2
sudo apt install ffmpeg -y
echo "Setting CONFIG ENV variable..."
echo "export CONFIG_PATH=$(readlink -f config.yaml)" >> ~/.bashrc
mkdir .cache
python3 -m venv .venv
source .venv/bin/activate
TMPDIR=.cache CMAKE_ARGS="-DLLAMA_NATIVE=off" pip install llama-cpp-python==0.2.78 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
TMPDIR=.cache pip3 install wheel webrtcvad --cache-dir .cache
TMPDIR=.cache pip install numpy==1.26.0 --cache-dir .cache
TMPDIR=.cache pip install simatic-3.1.2.tar.gz --cache-dir .cache --extra-index-url https://download.pytorch.org/whl/cpu --ignore-installed PyYAML
TMPDIR=.cache pip install scalene --cache-dir .cache
TMPDIR=.cache pip install nltk==3.8.1 --cache-dir cache
python3 -m nltk.downloader punkt wordnet stopwords punkt_tab