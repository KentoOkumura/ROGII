# Kaggle Platform — Comprehensive Knowledge Reference

> Official documentation: https://www.kaggle.com/docs
>
> Content review date: 2026-08-12.
>
> This reference summarizes all major Kaggle docs subpages so that any LLM
> using the KLLM skill has expert-level Kaggle knowledge in context.
> Platform capabilities, UI labels, competition rules, submission limits,
> storage limits, progression requirements, identity verification, package
> versions, hardware, quota, session limits, and accelerator types can change.
> Treat this file as dated background navigation, not as the operational source
> for those values. Check the live CLI, notebook metadata, runtime, competition
> rules, and official documentation immediately before acting.

## Documentation Map

| Page | URL | Summary |
|------|-----|---------|
| Main docs | https://www.kaggle.com/docs | Landing page — links to all sections |
| Competitions | https://www.kaggle.com/docs/competitions | Competition types, formats, submissions, leaderboards, teams, medals |
| Competitions Setup | https://www.kaggle.com/docs/competitions-setup | How to host/configure competitions |
| Datasets | https://www.kaggle.com/docs/datasets | Creating, versioning, downloading, and sharing datasets |
| Notebooks | https://www.kaggle.com/docs/notebooks | Cloud Jupyter environment, hardware, quotas, Docker images |
| Models | https://www.kaggle.com/docs/models | Model hierarchy, frameworks, publishing, downloading |
| API | https://www.kaggle.com/docs/api | Public API / CLI reference |
| MCP Server | https://www.kaggle.com/docs/mcp | MCP endpoint for AI agents |
| Organizations | https://www.kaggle.com/docs/organizations | Organization profiles and member management |
| Packages | https://www.kaggle.com/docs/packages | Pre-installed packages and custom installs |
| TPU | https://www.kaggle.com/docs/tpu | Kaggle TPU documentation; this repository's notebook generation does not support TPU |
| Efficient GPU | https://www.kaggle.com/docs/efficient-gpu-usage | GPU tips, mixed precision, gradient checkpointing |

---

## 1. Competitions (https://www.kaggle.com/docs/competitions)

### Competition Types

Kaggleのcompetition categories、公開状態、参加条件、賞品は変更され得る。
現在の分類と対象competitionの条件はcompetition一覧とrulesで確認する。

### Competition Formats

- **Standard:** Download data → build model locally → upload predictions CSV.
- **Code (Notebook-only):** All submissions from Kaggle Notebooks. Same hardware for all. Runtime limits, possible internet restrictions.
- **Two-Stage:** Stage 2 releases new test data; must submit in Stage 1 to access Stage 2.

### Submissions

- Evaluated by the competition's scoring metric (RMSE, AUC, F1, LogLoss, etc.).
- Daily submission limitと失敗提出の扱いは対象competitionのrulesとUIで確認する。
- Must accept competition rules before downloading data or submitting.

### Leaderboard System (Dual)

- **Public Leaderboard:** Scores on a subset of test data. Visible during competition.
- **Private Leaderboard:** Scores on the remaining test data. Determines final ranking after deadline. Overfitting risk between public and private.

### Teams

- Max team size varies per competition (defined in rules).
- **Team merging:** Allowed if merger deadline has not passed and merged size ≤ max.
- Teams can only disband if no submissions have been made.
- **Private sharing** of code/data outside teams is prohibited — all sharing must be public.

### Medals & Progression

Medal thresholdsとprogression requirementsは固定値として保存しない。
[Kaggle progression](https://www.kaggle.com/progression)で現在値を確認する。

---

## 2. Competitions Setup (https://www.kaggle.com/docs/competitions-setup)

### Creating a Competition

Navigate to https://www.kaggle.com/competitions/new. Types: Community (free, self-service), InClass (educators), Featured/Sponsored (paid).

### Configuration

- **Overview/Description:** Problem statement, goals, evaluation criteria, timeline.
- **Data:** Upload train.csv, test.csv, sample_submission.csv. For code competitions, data at `/kaggle/input/`.
- **Evaluation Metric:** The heart of the competition. Choose metric (RMSE, AUC, F1, etc.), upload solution file, set public/private split ratio. Custom metrics supported.
- **Scoring & Team Settings:** Max team size, daily submission limit, final submission selection.
- **Access:** Anyone / only with link / restricted email list (e.g., `@school.edu`).
- **Code Competition Toggle:** "Enable Notebooks and Models" — all submissions via Kaggle Notebooks.

### Public/Private Split

The solution file is split into public (shown during competition) and private (final ranking). Hosts can set the ratio or use a "Usage" column with "Private"/"Public" values.

---

## 3. Datasets (https://www.kaggle.com/docs/datasets)

### Creating Datasets

- **Web UI:** Datasets → New Dataset → drag-and-drop → set title, license, visibility.
- **CLI:** `kaggle datasets init -p dir` → edit metadata → `kaggle datasets create -p dir --dir-mode zip`.
- **kagglehub:** `kagglehub.dataset_upload("user/slug", "./dir", license_name="CC0-1.0")`.
- Sources: local files, remote URLs, GitHub repos, notebook outputs. Cannot mix source types.

### Metadata (`dataset-metadata.json`)

Follows Data Package spec. Required fields: `title`, `id` (`username/slug`), `licenses`. Optional: `subtitle` (20-80 chars), `keywords` (existing Kaggle tags), `resources` (file descriptions + schemas).

### Versioning

`kaggle datasets version -p dir -m "message"` — the `id` in metadata must match existing dataset.

### Size Limits

dataset size、private storage、per-file uploadの上限は固定値として保存しない。
upload直前にdataset docs、UI、利用中のCLIで現在値を確認する。

### Available Licenses

選択可能なlicenseはdataset作成時のUIと公式docsで確認する。licenseの意味や
適用可否はKaggleの選択肢だけから判断しない。

### Progression

現在の要件は[Kaggle progression](https://www.kaggle.com/progression)で確認する。

---

## 4. Notebooks (https://www.kaggle.com/docs/notebooks)

### Runtime resources and limits

Available CPU, memory, disk, accelerator types, session limits, access
requirements, and weekly quota are mutable platform settings. Before execution:

1. Inspect the generated `kernel-metadata.json` to identify the requested resource
   and verify that `enable_tpu` is `false`. Stop if TPU is requested because this
   repository's notebook generation and metadata validation do not support it.
2. For GPU, run `uv run kaggle quota --format json` and compare the live remaining
   time and refresh time with the expected notebook runtime.
3. Inspect the actual runtime for device and memory details, and check the
   official notebook documentation or UI for limits not exposed by the CLI.

Do not copy a fixed quota or hardware value from this reference into an
operational decision or experiment record.

### Languages & Docker

- **Python:** Docker image `kaggle/python` (GitHub: Kaggle/docker-python).
- **R:** Docker image `kaggle/rstats` (GitHub: Kaggle/docker-rstats).
- Kernel types: `notebook` (Jupyter) or `script` (standalone).

### Custom Packages

- **With internet:** `!pip install <package>` in a cell.
- **Without internet (offline competitions):** Download `.whl` files locally, upload as dataset, install from `/kaggle/input/`: `!pip install --no-index --find-links /kaggle/input/wheels/ package`.

### Saving

- **Auto-save:** Edits saved to draft automatically.
- **Save & Run All (Commit):** Runs all cells, saves output, creates a version.
- Only `/kaggle/working` persists. Content outside is lost on session end.
- Collaborators cannot edit simultaneously (unlike Google Colab).

### Data Access

- Competition data at `/kaggle/input/`.
- Attach datasets, models, or notebook outputs via "Add Data" sidebar.

### Progression

現在の要件は[Kaggle progression](https://www.kaggle.com/progression)で確認する。

---

## 5. Models (https://www.kaggle.com/docs/models)

### Hierarchy

```
Model → Framework → Variation → Version
```

Handle format: `owner/model/framework/variation` (e.g., `google/gemma/transformers/2b`).
Notebook path: `/kaggle/input/<model_slug>/<framework>/<variation>/<version>/`.

### Supported Frameworks

tensorFlow1, tensorFlow2, tfLite, tfJs, pyTorch, jax, coral, Keras.

### Publishing Models

1. **Web UI:** Models → New Model → fill metadata → upload files.
2. **kagglehub:** `kagglehub.model_upload("user/model/framework/variation", "./dir", license_name="Apache 2.0")`.
3. **CLI:** `kaggle models init` → `kaggle models create` → `kaggle models variations versions create`.
4. **KerasHub:** `keras_hub.upload_preset(uri="kaggle://user/model/Keras/variation", preset_dir="./dir")`.

### Metadata Files

- `model-metadata.json`: ownerSlug, title, slug, isPrivate, description (model card markdown), licenseName.
- `model-instance-metadata.json`: ownerSlug, modelSlug, instanceSlug, framework, overview, usage, licenseName, fineTunable, trainingData, modelInstanceType, baseModelInstance.

### Template Variables (for usage docs)

`${VERSION_NUMBER}`, `${VARIATION_SLUG}`, `${FRAMEWORK}`, `${PATH}`, `${FILEPATH}`, `${URL}`.

### Model Instance Types

- **Base Model:** Original standalone model.
- **Internal Variant:** Derived from another Kaggle model.
- **External Variant:** Derived from a model hosted elsewhere.

---

## 6. API / CLI (https://www.kaggle.com/docs/api, https://github.com/Kaggle/kaggle-cli)

### Installation

```bash
uv sync --locked    # Repository-pinned Kaggle CLI; requires Python 3.11+
```

### Authentication

認証方式、保存先、client別の対応、credential優先順位は[registrationの認証設定](../../registration/references/kaggle-setup.md)を正とする。ここには複製しない。

### Rate Limits

Kaggle uses dynamic rate limiting on both the API and website. If you get HTTP 429
("Too many requests"), wait a few minutes and retry. Check your code for unintended
loops or redundant calls.

### OAuth 2.0 Provider API

Kaggle implements OAuth 2.0 Authorization Code flow with PKCE for third-party apps:

**Endpoints:**
- Discovery: `GET https://www.kaggle.com/.well-known/oauth-authorization-server`
- Authorization: `GET https://www.kaggle.com/api/v1/oauth2/authorize`
- Token: `POST https://www.kaggle.com/api/v1/oauth2/token`
- Introspection: `POST https://www.kaggle.com/api/v1/oauth2/introspect`

**Scopes:** `datasets.get:*`, `datasets.create:*`, `datasets.update:*`, `models.get:*`, `models.download:*`, `kernels.list:*`, `kernels.pull:*`, `kernels.push:*`, `competitions.list:*`, `competitions.submit:*`.

**Roles (bundle permissions):** `datasets.viewer`, `datasets.editor`, `models.viewer`, `resources.admin`.

**Client types:** Public clients (PKCE required, localhost redirect only) and Organization clients (`org:<slug>`, HTTPS redirect allowed). Contact Kaggle team to register clients.

### Command Groups

```
kaggle auth         {login, print-access-token, revoke}
kaggle competitions {list, files, download, submit, submissions, leaderboard, episodes, replay, logs, pages, topics}
kaggle competitions topics {list, show}
kaggle datasets     {list, files, download, create, version, init, metadata, status, delete}
kaggle kernels      {list, files, init, push, pull, output, status, delete}
kaggle models       {list, get, init, create, update, delete}
kaggle models variations       {init, create, get, update, delete, files}
kaggle models variations versions {init, create, download, list, delete, files}
kaggle forums       {list, topics}
kaggle forums topics {list, show}
kaggle benchmarks   {auth, init, tasks, topics}
kaggle benchmarks tasks {push, run, list, status, download, log, models, delete, publish}
kaggle benchmarks topics {list, show}
kaggle config       {view, set, unset}
```

### Key Metadata Files

- **dataset-metadata.json:** title, id, licenses (required). Optional: subtitle, keywords, resources.
- **kernel-metadata.json:** id, title, code_file, language, kernel_type (required). Optional: is_private, enable_gpu, enable_internet, dataset_sources, competition_sources, model_sources.
- **model-metadata.json:** ownerSlug, title, slug, isPrivate, description, licenseName.
- **model-instance-metadata.json:** ownerSlug, modelSlug, instanceSlug, framework, overview, usage, licenseName.

### Available Accelerators (for `kaggle kernels push --accelerator`)

NvidiaTeslaP100, NvidiaTeslaT4, NvidiaTeslaT4Highmem, NvidiaTeslaA100, NvidiaL4, NvidiaL4X1, NvidiaH100, NvidiaRtxPro6000, TpuV38, Tpu1VmV38, TpuV5E8, TpuV6E8.

### Config Options

`kaggle config set -n {competition|path|proxy} -v VALUE`.

---

## 7. MCP Server (https://www.kaggle.com/docs/mcp)

### Endpoint

```
https://www.kaggle.com/mcp
```

Protocol: Streamable HTTP (MCP standard). Auth: API token via `Authorization: Bearer <api_token>` (use token from "Generate New Token" at kaggle.com/settings).

### Client Configuration

Use the MCP client's local secret store or protected environment-variable
reference for the Bearer token. Do not paste the token into a shell command,
command argument, committed configuration, or agent conversation. Repository
scripts read locally configured credentials and send the Authorization header
in-process through `shared/mcp_client.py`; follow
`references/mcp-reference.md` for the canonical helper setup.

### Available Tool Categories

- **Competitions:** list, details, download files, list files, submit, submissions, leaderboard.
- **Datasets:** list, list files, download, metadata, create new, create version, status, init metadata, update metadata.
- **Kernels/Notebooks:** list, list files, output, pull, status, init metadata, push.
- **Models:** list, get, init metadata, create, update, delete. Plus instance (variation) and version subtools.
- **Config:** view, set, unset, path.
- **Auth:** authenticate tool.

Use `tools/list` for discovery of exact tool names.

---

## 8. kagglehub (https://github.com/Kaggle/kagglehub)

### Installation

```bash
uv sync --locked --extra kaggle-platform
```

追加 adapter が必要なら、対象 extra を `pyproject.toml` に記録して lock file を更新してから同期する。

利用versionとPython要件は`uv.lock`と`pyproject.toml`を実行時に読み、ここへ固定値を転記しない。

### Authentication

`kagglehub.login()` (interactive), `KAGGLE_API_TOKEN` env var, `~/.kaggle/access_token`, `~/.kaggle/kaggle.json` (legacy), `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars (legacy), Google Colab secrets.

### All Functions

| Function | Purpose |
|----------|---------|
| `kagglehub.login()` | Interactive auth |
| `kagglehub.whoami()` | Show authenticated user |
| `kagglehub.dataset_download(handle, path=, force_download=, output_dir=)` | Download dataset |
| `kagglehub.dataset_upload(handle, local_dataset_dir, version_notes=, license_name=, ignore_patterns=)` | Upload/version dataset |
| `kagglehub.dataset_load(adapter, handle, path, pandas_kwargs=, sql_query=, hf_kwargs=, polars_frame_type=, polars_kwargs=)` | Load dataset into DataFrame |
| `kagglehub.model_download(handle, path=, force_download=, output_dir=)` | Download model |
| `kagglehub.model_upload(handle, local_model_dir, license_name=, version_notes=, ignore_patterns=, sigstore=)` | Upload/version model |
| `kagglehub.competition_download(handle, path=, force_download=, output_dir=)` | Download competition data |
| `kagglehub.notebook_output_download(handle, path=, force_download=, output_dir=)` | Download notebook output |

### Dataset Adapters

| Adapter | Returns | Formats |
|---------|---------|---------|
| `KaggleDatasetAdapter.PANDAS` | DataFrame | CSV, TSV, JSON, JSONL, XML, Parquet, Feather, SQLite, Excel |
| `KaggleDatasetAdapter.POLARS` | LazyFrame/DataFrame | CSV, TSV, JSON, JSONL, Parquet, Feather, SQLite, Excel |
| `KaggleDatasetAdapter.HUGGING_FACE` | HF Dataset | Same as Pandas (built on top) |

### Not Supported

- No kernel/notebook push, status, or output operations.
- No competition submit.
- No competition registration.
- No benchmark operations.

### Environment Variables

`KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLEHUB_CACHE` (default `~/.cache/kagglehub/`), `KAGGLE_CONFIG_DIR` (default `~/.kaggle/`), `KAGGLEHUB_VERBOSITY` (debug/info/warning/error/critical).

---

## 9. Organizations (https://www.kaggle.com/docs/organizations)

- Create at https://www.kaggle.com/organizations/new.
- Roles: **Admin** (full control) and **Member** (can contribute under org brand).
- Admins can invite/remove members, manage settings.
- Orgs can publish datasets, notebooks, and competitions under a shared identity.

---

## 10. Packages (https://www.kaggle.com/docs/packages)

- Kaggle notebooks come with hundreds of pre-installed packages.
- Docker images: `kaggle/python` (GitHub: Kaggle/docker-python) and `kaggle/rstats` (GitHub: Kaggle/docker-rstats).
- Package list: `kaggle_requirements.txt` in the docker-python repo.
- Custom packages: `!pip install X` (internet required). Offline: upload wheels as dataset.
- To run locally: `docker pull kaggle/python`.

---

## 11. TPU (https://www.kaggle.com/docs/tpu)

Kaggle platformのTPU仕様は公式資料で確認する。このリポジトリの
`prepare-kaggle-notebooks`とmetadata検証はTPUに対応せず、
`enable_tpu: true`を拒否する。TPU用コードやpackageをこのテンプレートから
生成できると解釈せず、必要な場合は未対応として停止する。

---

## 12. Efficient GPU Usage (https://www.kaggle.com/docs/efficient-gpu-usage)

### Runtime contract

Available GPU models, device count, VRAM, session limits, and weekly quota are
mutable. Inspect `kernel-metadata.json`, query `uv run kaggle quota --format json`,
and use `nvidia-smi` or the framework device API in the actual runtime. Do not
select an implementation from a fixed hardware list in this reference.

### Tips

1. **Mixed precision:** TF: `mixed_precision.set_global_policy('mixed_float16')`. PyTorch: `torch.cuda.amp.autocast()`.
2. **Batch size:** Powers of 2 (8, 16, 32...). For FP16, multiples of 8.
3. **Gradient accumulation:** Simulate larger batches without extra memory.
4. **Gradient checkpointing:** ~20% slower but large memory savings.
5. **Monitor:** `!nvidia-smi` or `pynvml`.
6. **Data loading:** `num_workers > 0`, `pin_memory=True` in PyTorch DataLoaders.
7. **Only enable GPU when needed** — turn off during exploration/preprocessing.

---

## 13. Progression System (https://www.kaggle.com/progression)

tiers、categories、medal thresholds、progression requirementsは変更され得る。
固定値をこのreferenceから使わず、上記公式ページで現在値を確認する。

---

## 14. Persona Identity Verification

identity verificationの要否、provider、必要書類、実行手順は変更され得る。
対象competitionのrulesとKaggle UIに表示される現在の案内に従い、本人確認情報を
エージェントへ渡さない。
