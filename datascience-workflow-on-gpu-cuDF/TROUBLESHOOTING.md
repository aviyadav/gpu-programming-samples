# Troubleshooting

## `uv add "polars[gpu]"` fails with `Didn't find wheel for cudf-cu12 24.8.3`

### Environment

| | |
|---|---|
| Python | CPython 3.14.2 |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 (x86_64) |
| Driver | 592.01 |
| CUDA | 13.1 |
| Package manager | uv |

### Symptom

```
$ uv add "polars[gpu]"

× Failed to build `cudf-cu12==24.8.3`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `nvidia_stub.buildapi.build_wheel` failed (exit status: 1)

      RuntimeError: Didn't find wheel for cudf-cu12 24.8.3

      *******************************************************************************
      The installation of cudf-cu12 for version 24.8.3 failed.

      This is a special placeholder package which downloads a real wheel package
      from https://pypi.nvidia.com. If https://pypi.nvidia.com is not reachable, we
      cannot download the real wheel file to install.
      *******************************************************************************

hint: `cudf-cu12` (v24.8.3) was included because `datascience-workflow-on-gpu-cudf` (v0.1.0)
depends on `polars[gpu]>=1.43.2` (v1.43.2) which depends on `cudf-polars-cu12` (v24.8.0a281)
which depends on `cudf-cu12>=24.8.dev0, <24.9.dev0`
```

The `nvidia-stub` traceback is a **red herring**. The network is fine and
`pypi.nvidia.com` is reachable — the real failure is a version conflict that
happened earlier during resolution.

### Root cause

Two separate problems stacked on top of each other.

**1. The `gpu` extra is hardcoded to CUDA 12.**

`polars[gpu]` declares a dependency on `cudf-polars-cu12`. This machine runs
CUDA 13, and the project already depends on `cudf-cu13`. There is no `cu13`
variant of the extra — the CUDA version is baked into the package name, so the
extra can never select the right one.

**2. Current `cudf-polars` caps polars below the pinned version.**

`cudf-polars-cu12` / `cudf-polars-cu13` 26.8.0 both declare:

```
polars<1.43,>=1.35
```

while `pyproject.toml` required:

```
polars>=1.43.2
```

These are unsatisfiable together. uv backtracked through progressively older
`cudf-polars-cu12` releases looking for one without the upper bound, and
eventually reached `24.8.0a281` — an sdist placeholder published in 2024. That
placeholder builds by downloading a real wheel from `pypi.nvidia.com` at build
time, and no `cudf-cu12` 24.8.3 wheel exists for Python 3.14 (it predates 3.14
by over a year). Hence the misleading "didn't find wheel" error.

In short: **the polars upper bound was the actual problem; the stub build
failure was just the last symptom in a long backtrack.**

### Fix

Drop the `gpu` extra, constrain polars to the range RAPIDS supports, and depend
on the CUDA 13 package directly:

```sh
uv add "polars>=1.35,<1.43" "cudf-polars-cu13>=26.8.0"
```

Resulting entries in `pyproject.toml`:

```toml
dependencies = [
    "cudf-cu13>=26.8.0",
    "cudf-polars-cu13>=26.8.0",
    "polars>=1.35,<1.43",
]
```

This downgrades polars 1.43.2 → 1.42.1 and installs `cudf-polars-cu13`,
`cudf-streaming-cu13`, `rapidsmpf-cu13`, plus the ucx/libcudf runtime
dependencies. No NVIDIA index is needed — RAPIDS publishes real `cu13` wheels
to PyPI proper.

### Verification

```sh
uv run python -c "
import polars as pl
engine = pl.GPUEngine(device=0, raise_on_fail=True)
df = pl.LazyFrame({'a': [1, 2, 3, 4], 'b': ['x', 'y', 'x', 'y']})
print(df.group_by('b').agg(pl.col('a').sum()).sort('b').collect(engine=engine))
"
```

```
shape: (2, 2)
┌─────┬─────┐
│ b   ┆ a   │
│ --- ┆ --- │
│ str ┆ i64 │
╞═════╪═════╡
│ x   ┆ 4   │
│ y   ┆ 6   │
└─────┴─────┘
```

### Notes

- **Match the CUDA suffix to your toolkit.** Check with `nvidia-smi`. Use
  `cudf-polars-cu13` for CUDA 13, `cudf-polars-cu12` for CUDA 12. Never mix
  `cu12` and `cu13` packages in one environment.
- **Use `raise_on_fail=True` while developing.** By default polars silently
  falls back to CPU for unsupported operations, which looks like a working
  GPU query that is mysteriously slow:

  ```python
  engine = pl.GPUEngine(device=0, raise_on_fail=True)
  df.collect(engine=engine)
  ```

  Plain `collect(engine="gpu")` is fine for production runs where fallback is
  acceptable.
- **The `polars<1.43` cap blocks upgrades.** `uv lock --upgrade` will not move
  polars past 1.42.x until RAPIDS ships a release supporting 1.43+. Widen the
  bound once it does.
- **Reading a `Failed to build` error from uv:** when uv reports a build
  failure for a suspiciously old version of a package you never asked for,
  read the `hint:` line at the bottom first. It shows the dependency chain, and
  an ancient version there almost always means resolution backtracked into
  unusable territory because of a conflicting constraint elsewhere.
