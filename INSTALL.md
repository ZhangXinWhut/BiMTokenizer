# Environment setup

Python **3.10**, CUDA **12.8**, PyTorch **2.8.0**. One `requirements.txt` for all machines. `causal_conv1d` and `mamba_ssm` are compiled from GitHub source for the GPU you actually have. Both builds run the same BiMTokenizer code.

| GPU | Arch (`TORCH_CUDA_ARCH_LIST`) | causal-conv1d | mamba-ssm |
|-----|-------------------------------|---------------|-----------|
| NVIDIA H100 | `9.0` | [v1.5.2](https://github.com/Dao-AILab/causal-conv1d/tree/v1.5.2) | [v2.2.5](https://github.com/state-spaces/mamba/tree/v2.2.5) |
| GeForce RTX 5090 | `12.0` | [v1.5.4](https://github.com/Dao-AILab/causal-conv1d/tree/v1.5.4) | [v2.2.6](https://github.com/state-spaces/mamba/tree/v2.2.6) |

Do not `pip install causal-conv1d mamba-ssm` from PyPI: the wheel’s CUDA arch usually does not match. Clone, then `pip install .` with `*_FORCE_BUILD=TRUE`.

## 1. Conda env + Python deps

```bash
conda create -n bimtokenizer python=3.10 -y
conda activate bimtokenizer
pip install -r requirements.txt
```

`causal_conv1d` / `mamba_ssm` look like `file:///path/to/...` after a local source install; compile them with the scripts below instead of copying that path.

## 2. Compile CUDA extensions

H100 (this repo’s `zx` env):

```bash
bash install_causal_conv1d.sh h100
bash install_mamba_ssm.sh h100
```

RTX 5090:

```bash
bash install_causal_conv1d.sh 5090
bash install_mamba_ssm.sh 5090
```

The scripts clone into `third_party/` (or reuse `SRC_DIR` if you already cloned). Override CUDA toolkit with `CUDA_HOME=/usr/local/cuda-12.8` if needed. Compile jobs default to `MAX_JOBS=2`.

If the repo is already cloned:

```bash
# H100 example
SRC_DIR=/path/to/causal-conv1d-1.5.2 bash install_causal_conv1d.sh h100
SRC_DIR=/path/to/mamba-2.2.5         bash install_mamba_ssm.sh h100
```

## 3. Check

```bash
python -c "import torch, causal_conv1d, mamba_ssm; print(torch.cuda.get_device_name(0), causal_conv1d.__version__, mamba_ssm.__version__)"
```
