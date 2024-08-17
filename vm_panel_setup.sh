sudo apt update
sudo apt install build-essential cmake git python3-dev python3-pip -y
sudo apt install musl-dev -y
sudo ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1
sudo apt install python3-venv -y
cd /media/root-rw
mkdir .cache
python3 -m venv venv
source venv/bin/activate
TMPDIR=.cache CMAKE_ARGS="-DLLAMA_NATIVE=off" pip install llama-cpp-python==0.2.78 --force-reinstall --cache-dir .cache --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
TMPDIR=.cache pip install simatic-3.0.0.tar.gz --cache-dir .cache --extra-index-url https://download.pytorch.org/whl/cpu --ignore-installed PyYAML
TMPDIR=.cache pip install scalene --cache-dir .cache