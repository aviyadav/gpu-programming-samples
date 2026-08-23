# Data Science Workflow on GPU with cuDF

Accelerating a typical data-science workflow — specifically **data preparation** —
on an NVIDIA GPU using [RAPIDS cuDF](https://rapids.ai/) and the
[Polars GPU engine](https://docs.pola.ai/). All examples use the
**NYC Yellow Taxi** trip records as the working dataset.

The project explores three ways to move pandas-style work onto the GPU, from
"change nothing" to "fully GPU-native":

| Approach | Package | How it works |
|---|---|---|
| `cudf.pandas` accelerator | `cudf-cu13` | Runs *existing* pandas code on the GPU via a drop-in loader; unsupported ops fall back to CPU automatically. |
| cuDF API | `cudf-cu13` | Work directly with GPU DataFrames using a pandas-like API. |
| Polars GPU engine | `cudf-polars-cu13` | Express work as Polars lazy queries and run them on the GPU with `collect(engine="gpu")`. |

---

## Repository layout

```
.
├── pyproject.toml                          # Project metadata + dependencies (managed by uv)
├── uv.lock                                 # Locked dependency graph
├── load_data.py                            # Minimal cuDF smoke-test: read one taxi month from URL
├── TROUBLESHOOTING.md                      # Known errors + fixes (e.g. polars[gpu] on CUDA 13)
└── accelerating-data-preparation/
    ├── cudf_pandas_gpu_workflows.ipynb     # Accelerate existing pandas code with cudf.pandas
    ├── cudf_gpu_workflows.ipynb            # GPU-native workflows with the cuDF API + benchmarks
    └── polars_gpu_engine_workflows.ipynb   # Polars lazy API on the GPU engine, incl. fallback demo
```

### What each notebook covers

- **`cudf_pandas_gpu_workflows.ipynb`** — Load the taxi data, run a grouped
  summary written as ordinary pandas code, and profile GPU vs. CPU execution
  with `cudf.pandas` loaded.
- **`cudf_gpu_workflows.ipynb`** — Work directly with cuDF DataFrames: load
  ~9–10M rows across three months, everyday operations, moving data between
  pandas and cuDF, and `%time` benchmarks for grouped summaries and chained
  operations.
- **`polars_gpu_engine_workflows.ipynb`** — The same analysis as Polars lazy
  queries executed with `engine="gpu"`, CPU/GPU timing, verifying GPU
  execution with `pl.GPUEngine(raise_on_fail=True)`, and a deliberate demo of
  graceful CPU fallback for unsupported operations.

---

## Requirements

- **NVIDIA GPU** with a CUDA 13-capable driver. Check with:

  ```sh
  nvidia-smi
  ```

- **CUDA 13.x** runtime. This project pins the `*-cu13` RAPIDS wheels
  (`cudf-cu13`, `cudf-polars-cu13`). Do not mix `cu12` and `cu13` packages.
- **Python >= 3.14** (see `requires-python` in `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** for environment and dependency management.

> **Note:** RAPIDS wheels are published for Linux x86_64 / aarch64. On Windows
> use WSL2 (this project was developed in WSL2).

---

## Setup

Install `uv` if you don't have it:

```sh
# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via pip
pip install uv
```

Then, from the project root, create the virtual environment and install the
locked dependencies:

```sh
uv sync
```

This reads `pyproject.toml` / `uv.lock`, creates `.venv/`, and installs
everything needed to run the notebooks and scripts.

---

## Build & run

Python projects don't have a separate compile step, so **"build" = resolving
and installing the environment**, and **"run" = launching the notebooks or a
script** through `uv run`.

### Build / install the environment

```sh
# Install all dependencies from the lockfile into .venv
uv sync

# Re-lock after editing pyproject.toml
uv lock

# Add a new dependency (locks + installs). Match the CUDA suffix to your toolkit.
uv add cudf-polars-cu13
```

### Run

```sh
# Quick GPU smoke test: reads one month of taxi data with cuDF
uv run load_data.py

# Launch Jupyter and open the notebooks in accelerating-data-preparation/
uv run jupyter lab

# Run a notebook non-interactively (example)
uv run jupyter nbconvert --to notebook --execute \
  accelerating-data-preparation/polars_gpu_engine_workflows.ipynb
```

All commands are run from the project root so `uv` picks up the local `.venv`.

---

## Dependencies

Declared in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `cudf-cu13` | GPU DataFrame library (cuDF) + the `cudf.pandas` accelerator, CUDA 13 build. |
| `cudf-polars-cu13` | GPU engine backend for Polars lazy queries, CUDA 13 build. |
| `polars` (>=1.35,<1.43) | DataFrame / lazy query engine. Pinned below 1.43 to match the current `cudf-polars` release. |
| `pandas` | Reference CPU implementation for comparisons. |
| `jupyter` | Notebook server for the walkthrough notebooks. |
| `plotly` | Interactive plotting. |
| `aiohttp` | Async HTTP (used for data download helpers). |
| `pip` | Retained for interoperability with pip-based tooling. |

### Data source

Notebooks pull the public NYC Yellow Taxi parquet files, e.g.:

```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
```

An internet connection is required the first time a notebook downloads data.

---

## Using the GPU engines

```python
# cudf.pandas: run existing pandas code on the GPU
# (enable via the `%load_ext cudf.pandas` magic in a notebook,
#  or `python -m cudf.pandas script.py` on the command line)

import cudf
df_gpu = cudf.read_parquet(url)          # GPU-native cuDF API

import polars as pl
q = df.lazy().group_by("PULocationID").agg(pl.len())
q.collect(engine="gpu")                  # Polars GPU engine

# Strict mode: error instead of silently falling back to CPU
engine = pl.GPUEngine(device=0, raise_on_fail=True)
q.collect(engine=engine)
```

---

## Troubleshooting

See [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) for recorded errors and their
fixes, including:

- Why `uv add "polars[gpu]"` fails on CUDA 13 and how to fix it (use
  `cudf-polars-cu13` and pin `polars<1.43`).
- Reading uv build-failure `hint:` lines when resolution backtracks into
  placeholder packages.
