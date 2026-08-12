# kagglehub API Reference

> Official source: https://github.com/Kaggle/kagglehub
> PyPI: https://pypi.org/project/kagglehub/
> Repository-locked version and Python requirements: read `uv.lock` and `pyproject.toml` at execution time; do not transcribe fixed values here.

## Installation

```bash
uv sync --locked --extra kaggle-platform
```

`pandas-datasets`、`polars-datasets`、`hf-datasets`、`signing` の adapter が必要な場合は、必要な extra を `pyproject.toml` の `kaggle-platform` 依存へ追加して `uv lock` を更新し、その後 `uv sync --locked --extra kaggle-platform` を実行する。セッション内だけの install は行わない。

## Authentication

credentialの生成、保存、優先順位は[registrationの認証設定](../../registration/references/kaggle-setup.md)を正とする。以下はkagglehub固有のAPIだけを示す。

```python
import kagglehub

# Option 1: Interactive login
kagglehub.login()

# Option 2: Programmatic
from kagglehub.config import set_kaggle_credentials, set_kaggle_api_token
set_kaggle_credentials(username="...", api_key="...")
set_kaggle_api_token(api_token="...")

# Check who is logged in
kagglehub.whoami()  # Returns: {'username': '...'}
```

Inside Kaggle notebooks, authentication is automatic.

## All Functions

### dataset_download()

```python
kagglehub.dataset_download(
    handle: str,               # "owner/dataset" or "owner/dataset/versions/N"
    path: str | None = None,   # specific file within dataset
    force_download: bool = False,
    output_dir: str | None = None,  # custom dir (bypasses cache)
) -> str  # returns local path
```

### dataset_upload()

```python
kagglehub.dataset_upload(
    handle: str,                  # "owner/dataset" (no version)
    local_dataset_dir: str,
    version_notes: str = "",
    ignore_patterns: list[str] | str | None = None,
) -> None
```

Creates dataset if new; creates new version if exists. Default ignore: `.git/`, `.cache/`, `.huggingface/`.

**Note:** Unlike `model_upload()`, `dataset_upload()` does NOT accept a `license_name` parameter.

### dataset_load()

```python
kagglehub.dataset_load(
    adapter: KaggleDatasetAdapter,  # PANDAS, POLARS, or HUGGING_FACE
    handle: str,
    path: str,                      # file within dataset
    pandas_kwargs: Any = None,      # passed to pandas read_* method
    sql_query: str | None = None,   # for SQLite files
    hf_kwargs: Any = None,          # passed to Dataset.from_pandas()
    polars_frame_type: PolarsFrameType | None = None,  # LAZY_FRAME or DATA_FRAME
    polars_kwargs: Any = None,
) -> DataFrame | LazyFrame | Dataset
```

**Adapters:**
| Adapter | Returns | Install Extra |
|---------|---------|---------------|
| `KaggleDatasetAdapter.PANDAS` | pandas DataFrame | `[pandas-datasets]` |
| `KaggleDatasetAdapter.POLARS` | polars LazyFrame (default) or DataFrame | `[polars-datasets]` |
| `KaggleDatasetAdapter.HUGGING_FACE` | HF Dataset (via pandas) | `[hf-datasets]` |

**Supported formats:** CSV, TSV, JSON, JSONL, XML, Parquet, Feather, SQLite, Excel.

### model_download()

```python
kagglehub.model_download(
    handle: str,               # "owner/model/framework/variation" or with /version
    path: str | None = None,   # specific file
    force_download: bool = False,
    output_dir: str | None = None,
) -> str
```

### model_upload()

```python
kagglehub.model_upload(
    handle: str,                  # "owner/model/framework/variation" (no version)
    local_model_dir: str,
    license_name: str | None = None,  # e.g. "Apache 2.0"
    version_notes: str = "",
    ignore_patterns: list[str] | str | None = None,
    sigstore: bool = False,       # requires kagglehub[signing]
) -> None
```

### competition_download()

```python
kagglehub.competition_download(
    handle: str,               # competition slug
    path: str | None = None,
    force_download: bool = False,
    output_dir: str | None = None,
) -> str
```

### notebook_output_download()

```python
kagglehub.notebook_output_download(
    handle: str,               # "owner/notebook-slug"
    path: str | None = None,
    force_download: bool = False,
    output_dir: str | None = None,
) -> str
```

### login() / whoami()

```python
kagglehub.login(validate_credentials: bool = True) -> None
kagglehub.whoami(verbose: bool = True) -> dict  # {'username': '...'}
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `KAGGLE_API_TOKEN` | API token | — |
| `KAGGLE_USERNAME` | Legacy username | — |
| `KAGGLE_KEY` | Legacy API key | — |
| `KAGGLEHUB_CACHE` | Cache folder | `~/.cache/kagglehub/` |
| `KAGGLE_CONFIG_DIR` | Credentials folder | `~/.kaggle/` |
| `KAGGLEHUB_VERBOSITY` | Log level (debug/info/warning/error/critical) | `info` |

## Not Supported

- No kernel/notebook push, status, or output operations
- No competition submit
- No competition registration
- No benchmark operations
