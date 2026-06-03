# CuTeGen

## Setup
Install Miniconda from https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation
```bash
conda create --name cutegen python=3.10
conda activate cutegen
pip install -r requirements.txt
pip install -e .
```

NOTE: If you are using newer versions of CUDA (>= 12.8), and the above doesn't work, try the following:
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Some environment variables to set
- `CUTEGEN_BASE_PATH` should be an absolute path to this repo, cutegen.
- Relevant API keys depending on the models you use: `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `SGLANG_API_KEY`, `ANTHROPIC_API_KEY`, `SAMBANOVA_API_KEY`, `FIREWORKS_API_KEY`
- `NSIGHT_COMPUTE_BIN` should be an absolute path to the Nsight Compute (ncu) executable (e.g. export NSIGHT_COMPUTE_BIN=/usr/local/cuda/bin/ncu)
- `CUDACXX` should be an absolute path to the nvcc compiler (e.g. `export CUDACXX=/usr/local/cuda/bin/nvcc`) 
- `CUTLASS_BASE_PATH` and `CUTLASS_INCLUDE_PATH` should be an absolute path to the CUTLASS repo and its include directory

### Set up NCU profiler
If it's not already installed, install it from https://developer.nvidia.com/tools-overview/nsight-compute/get-started
If it's already installed, it's usually in `/usr/local/cuda/bin/`'

### Set up CUTLASS 
Make sure you have a complete cudatoolkit installed: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_local
Then install CUTLASS: https://github.com/NVIDIA/cutlass

## Running CuTeGen
1. Change the settings in `cutegen/config.py`
2. Change the ref files in `cutegen/main.py`
3. Run the main script
```
python cutegen/main.py
```
