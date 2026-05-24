# PyTorch

PyTorch model training experiments and implementations. Covers the core training loop, custom datasets, CNNs, and GPU-accelerated training with automatic CPU fallback.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)
![Jupyter](https://img.shields.io/badge/JupyterLab-3.x-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio jupyterlab matplotlib
jupyter lab
```

GPU training (optional — falls back to CPU automatically):

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Contents

| File / Notebook | Description |
|---|---|
| `train.py` | Training loop: DataLoader, loss, optimizer, eval |
| `models/` | Model definitions (CNN, MLP, ResNet-style) |
| `datasets/` | Custom `Dataset` and `DataLoader` examples |
| `notebooks/` | Exploration and visualization notebooks |

---

## Training

```bash
python train.py --epochs 10 --batch-size 64 --lr 1e-3
```

Device detection is automatic:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

## Requirements

- Python 3.9+
- PyTorch 2.x
- torchvision
- JupyterLab (for notebooks)
