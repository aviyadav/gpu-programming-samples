# cuda-python-ex

A minimal example of running a CUDA kernel from Python using [numba](https://numba.readthedocs.io/).
The kernel adds two arrays element-wise on the GPU.

```
Array a :  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
Array b :  [  0.  12.  24.  36.  48.  60.  72.  84.  96. 108.]
Result a + b :  [  0.  13.  26.  39.  52.  65.  78.  91. 104. 117.]
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Linux (x86-64) | Tested on Ubuntu 24.04+ / WSL2 |
| NVIDIA GPU | Driver ≥ 525, CUDA capability ≥ 5.0 |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

> **No system CUDA toolkit needed.** The required libraries are pulled in as
> Python packages.

---

## Quick start

```bash
uv sync
bash setup-cuda.sh
uv run main.py
```

`setup-cuda.sh` only needs to be run once (or again after deleting `.venv`).

---

## Why a setup script is needed (conda vs uv)

The original example was written for **conda**, where
`conda install cudatoolkit` drops the real CUDA libraries
(`libnvvm.so`, `libcudart.so`, `libdevice.10.bc`) into the conda environment
and numba finds them automatically.

With **uv** / pip there is no `cudatoolkit` equivalent.
The `cuda-toolkit` package on PyPI is an empty stub with no binaries.
The actual libraries must come from NVIDIA's own PyPI packages:

| PyPI package | Provides |
|---|---|
| `nvidia-cuda-nvcc-cu12` | `libnvvm.so`, `libdevice.10.bc` |
| `nvidia-cuda-runtime-cu12` | `libcudart.so.12` |

These are declared in `pyproject.toml` and installed by `uv sync`, but numba
cannot find them automatically because it only searches a handful of
well-known paths (conda env, `CUDA_HOME`, `/usr/local/cuda`).
`setup-cuda.sh` bridges this gap with two steps described below.

---

## What setup-cuda.sh does

### Step 1 — Build a `CUDA_HOME` staging directory

numba's path-discovery code (`numba/cuda/cuda_paths.py`) looks for libraries
inside a `CUDA_HOME` directory with this layout:

```
$CUDA_HOME/
  nvvm/
    lib64/
      libnvvm.so.4      ← versioned name required by numba's find_lib()
    libdevice/
      libdevice.10.bc
  lib64/
    libcudart.so.12
```

The script creates `.venv/cuda_home/` and populates it with symlinks into
the PyPI packages:

```
.venv/cuda_home/nvvm/lib64/libnvvm.so.4
  → .venv/.../nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so

.venv/cuda_home/nvvm/libdevice/libdevice.10.bc
  → .venv/.../nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc

.venv/cuda_home/lib64/libcudart.so.12
  → .venv/.../nvidia/cuda_runtime/lib/libcudart.so.12
```

Two quirks worth noting:

- **Versioned symlink** — the PyPI package ships `libnvvm.so` (no version
  suffix). numba's `find_lib()` uses a regex that requires `libnvvm.so.N`, so
  an extra `libnvvm.so.4` symlink is created alongside the unversioned one.
- **Separate runtime** — `libcudart.so.12` lives in a different PyPI package
  from `libnvvm.so`, so both packages are needed and their libraries land in
  different locations.

### Step 2 — Install `sitecustomize.py`

uv doesn't load `.env` files automatically, so `CUDA_HOME` cannot be set
via a project config file alone.

The script writes a `sitecustomize.py` into the venv's `site-packages`.
Python executes this file automatically at startup, before any user code
runs, making `CUDA_HOME` available to numba without any extra flags:

```python
# .venv/lib/pythonX.Y/site-packages/sitecustomize.py
import os, pathlib

_cuda_home = pathlib.Path(__file__).parent.parent.parent.parent / "cuda_home"
if _cuda_home.exists() and "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = str(_cuda_home)
```

After this, `uv run main.py` works with no extra environment variables.

---

## Changes to the original program

The original example was written for an older numba API. One change was
required:

### `DeviceNDArray.free()` removed

Newer numba removed the `.free()` method on device arrays.
Use `del` instead to release GPU memory explicitly, or simply let the
garbage collector handle it.

```python
# Before (original — raises AttributeError on numba ≥ 0.57)
a_gpu.free()
b_gpu.free()
result_gpu.free()

# After
del a_gpu
del b_gpu
del result_gpu
```

---

## GPU architecture note

numba 0.65 knows compute capabilities up to **sm_90** (Hopper / RTX 4000
series). If your GPU is newer — for example an RTX 5000 series card with
**sm_120** (Blackwell) — numba falls back to compiling for sm_90, which runs
correctly via CUDA's forward compatibility.

A future numba release will add native sm_120 support.
If you need to pin a specific target CC in the meantime, set:

```bash
NUMBA_CUDA_DEFAULT_PTX_CC=9.0 uv run main.py
```

---

## Project structure

```
cuda-python-ex/
├── main.py              # GPU kernel + host code
├── pyproject.toml       # dependencies (includes nvidia-cuda-* packages)
├── setup-cuda.sh        # one-time post-sync setup script
├── uv.lock
└── .venv/               # created by uv sync (gitignored)
    ├── cuda_home/       # staging CUDA_HOME created by setup-cuda.sh
    │   ├── lib64/
    │   │   └── libcudart.so.12 -> ...
    │   └── nvvm/
    │       ├── lib64/
    │       │   ├── libnvvm.so -> ...
    │       │   └── libnvvm.so.4 -> ...
    │       └── libdevice/
    │           └── libdevice.10.bc -> ...
    └── lib/pythonX.Y/site-packages/
        ├── sitecustomize.py   # auto-sets CUDA_HOME at startup
        ├── nvidia/
        │   ├── cuda_nvcc/     # from nvidia-cuda-nvcc-cu12
        │   └── cuda_runtime/  # from nvidia-cuda-runtime-cu12
        └── numba/
```
