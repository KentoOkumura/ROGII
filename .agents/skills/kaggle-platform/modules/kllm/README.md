# KLLM — Kaggle Interaction Module

Interact with kaggle.com using kagglehub, Kaggle CLI v2.2.3, Kaggle MCP
Server, or Kaggle UI. Credential storage and priority follow the canonical
[registration guide](../registration/references/kaggle-setup.md). Do not create
a project `.env` solely for Kaggle credentials. **Never put credential values
in command arguments, logs, displayed output, or committed files**.

## Credentials

Before an authenticated operation, verify the credential required by the
selected client without exposing its value:

```bash
# Local Kaggle CLI operations
uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require cli

# Kaggle Python API and kagglehub operations
uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require python-api

# Kaggle MCP Server operations
uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require api-token
```

認証方式、保存先、優先順位は[registrationの認証設定](../registration/references/kaggle-setup.md)を正とする。CLIはOAuth、API token、legacy username/key、kagglehubはAPI tokenまたはlegacy username/key、MCPはAPI tokenを使う。tokenの種類をprefixから推測しない。

## Four Methods of Interaction

| Method | Type | Best For |
|--------|------|----------|
| **kagglehub** | Python library（`uv sync --locked --extra kaggle-platform`） | Quick dataset/model download in Python |
| **kaggle-cli** | CLI（`uv sync --locked`） | Full workflow scripting (competitions, notebooks, datasets, models, forums/topics, benchmarks) |
| **Kaggle MCP Server** | Remote endpoint `https://www.kaggle.com/mcp` | AI agent integration (Claude Code, gemini-cli, Cursor, etc.) |
| **Kaggle UI** | ユーザーが操作するbrowser | Account setup, verification, visual exploration |

## Capability Matrix

| Task | kagglehub | kaggle-cli | MCP Server | UI |
|------|-----------|------------|------------|-----|
| Search competitions | — | `kaggle competitions list` | `search_competitions` | Yes |
| Get competition metadata | — | — | `get_competition` | Yes |
| Read competition overview pages (rules / evaluation / data-description / FAQ / prizes / timeline) | — | — | `list_competition_pages` ([guide](references/competition-overview.md)) | Yes |
| Read competition discussion topics | — | `kaggle competitions topics list/show` | `list_forum_topics` / `get_forum_topic` | Yes |
| List competition data files | — | `kaggle competitions files` | `list_competition_data_files` / `list_competition_data_tree_files` / `get_competition_data_files_summary` | Yes |
| Download competition data | `competition_download()` | `kaggle competitions download` | `download_competition_data_file` / `download_competition_data_files` | Yes |
| Submit a file to a standard competition (not this repository) | — | `kaggle competitions submit` | `start_competition_submission_upload` → `submit_to_competition` | Yes |
| Submit a notebook to this code competition | — | repository task `task submit-code` | `create_code_competition_submission` 🔒 | Yes |
| List/search submissions | — | `kaggle competitions submissions` | `search_competition_submissions` / `get_competition_submission` | Yes |
| Read leaderboard | — | `kaggle competitions leaderboard` | `get_competition_leaderboard` / `download_competition_leaderboard` | Yes |
| Search datasets | — | `kaggle datasets list` | `search_datasets` | Yes |
| Get dataset info / metadata | — | `kaggle datasets metadata` | `get_dataset_info` / `get_dataset_metadata` / `get_dataset_files_summary` / `get_dataset_status` | Yes |
| List dataset files | — | `kaggle datasets files` | `list_dataset_files` / `list_dataset_tree_files` | Yes |
| Download dataset | `dataset_download()` | `kaggle datasets download` | `download_dataset` | Yes |
| Upload dataset file / version | `dataset_upload()` | `kaggle datasets create` / `kaggle datasets version` | `upload_dataset_file` / `update_dataset_metadata` | Yes |
| Search notebooks | — | `kaggle kernels list` | `search_notebooks` | Yes |
| Get notebook info | — | — | `get_notebook_info` / `list_notebook_files` | Yes |
| Execute a Kaggle Notebook | — | `kaggle kernels push/logs -f/output` | `create_notebook_session` → `get_notebook_session_status` → `download_notebook_output[_zip]` | Yes |
| Cancel notebook session | — | — | `cancel_notebook_session` | Yes |
| Save / version notebook | — | `kaggle kernels push` | `save_notebook` | Yes |
| List models | — | `kaggle models list` | `list_models` | Yes |
| Get model + variations + versions | — | — | `get_model` / `list_model_variations` / `get_model_variation` / `list_model_variation_versions` / `list_model_variation_version_files` | Yes |
| Download model variation version | `model_download()` | `kaggle models variations versions download` | `download_model_variation_version` | Yes |
| Create / update model + variation | — | `kaggle models create` | `create_model` / `update_model` / `update_model_variation` | Yes |
| Forums (list / topics / threads) | — | `kaggle forums list`, `kaggle forums topics list/show` | `list_forums` / `list_forum_topics` / `get_forum` / `get_forum_topic` | Yes |
| Hackathon overview | — | — | `get_hackathon_overview` ([guide](hackathon/references/hackathon-endpoints.md)) | Yes |
| Hackathon writeups (list / fetch / by topic / by slug) | — | — | `list_hackathon_write_ups` / `get_writeup` / `get_writeup_by_topic` / `get_writeup_by_slug` / `get_resolved_writeup_links` | Yes |
| Hackathon tracks | — | — | `list_hackathon_tracks` | Yes |
| Hackathon writeup CSV export (host/judge) | — | — | `download_hackathon_write_ups` 🔒 | Yes |
| Benchmarks | — | `kaggle benchmarks auth/init/tasks` | `create_benchmark_task_from_prompt` / `get_benchmark_leaderboard` ([guide](hackathon/references/benchmark-endpoints.md)) | Yes |
| Episodes (simulation logs / replays) | — | — | `get_episode_agent_logs` / `get_episode_replay` / `list_submission_episodes` ([guide](hackathon/references/episode-endpoints.md)) | Yes |
| Authorize / user profile | — | — | `authorize` / `get_user_profile` | Yes |
| Generic search | — | — | `search_content` | Yes |
| Register account | — | — | — | UI only |
| Get API tokens | — | — | — | UI only |
| Persona verification | — | — | — | UI only |

🔒 = role-gated. See [mcp-reference.md](references/mcp-reference.md) for the
dated inventory observed on 2026-04-22 and retested on 2026-05-04. Confirm the
current tools with `tools/list` at runtime. The `kagglehub` and `kaggle-cli`
columns are deliberately sparse — most workflows are now better served via
the bundled MCP server.

## Known Issues

- **`dataset_load()` in kagglehub**: was broken in v0.4.3 (404 on
  `DownloadDataset`). Read the current repository version from `uv.lock`; the
  historical v0.4.3 result does not establish that version's runtime status.
  Test the target dataset first and, if it fails, fall back to
  `dataset_download()` + `pd.read_csv()` on the cached files.
- **`competitions download` does not support `--unzip`** in kaggle CLI >= 1.8.
  Only `datasets download` supports `--unzip`. Unzip competition data manually
  after a direct CLI download. In this repository, prefer `task dl-kaggle-comp`;
  it downloads the archive, rejects path traversal and symbolic links, safely
  extracts into `data.raw_dir`, skips only size-and-checksum-identical files, and
  stops instead of overwriting a differing existing file.
- **Competition-linked datasets** (e.g., `titanic/titanic`) return 403 even
  with valid credentials. Use standalone dataset copies or download via
  `competitions download`.
- **`competitions topic-messages` is deprecated** in CLI v2.2.0+. Use
  `kaggle competitions topics show COMPETITION TOPIC_ID` for discussion content.
- **`competition_download()` 401 in kagglehub** (older versions): same
  caveat as `dataset_load()` — verify the version in the current `uv.lock` at runtime.
  For "rules not accepted" errors, navigate to
  `https://www.kaggle.com/competitions/<slug>/rules` in the browser and click accept.
- **MCP Server auth**: Use the API token from "Generate New Token" at
  kaggle.com/settings. Token形式やendpoint別の可否をprefixから推測せず、
  task-relevant authenticationをlive endpointで確認する。
- **Rate limiting**: Kaggle uses dynamic rate limiting. If you get HTTP 429,
  wait a few minutes and retry. Check code for unintended loops or redundant
  API calls.
- **`get_hackathon_write_up`**: was returning generic invocation errors in
  the kmcp-tools 2026-04-22 audit; the 2026-05-04 retest returned PASS.
  Treat this as dated evidence and recheck the live endpoint when using it.
  The hackathon submodule's `fetch_writeup.py` uses `get_writeup` first
  anyway because it has a simpler arg shape.

## Task Workflows

### Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("owner/dataset-name")
```
```bash
bash .agents/skills/kaggle-platform/modules/kllm/scripts/cli_download.sh owner/dataset-name
```

外部 Dataset をリポジトリ内へ保存する場合は `data/external/<owner-dataset>/` を使う。上の同梱 script もそこを既定値とする。公式コンペデータは同じ場所へ混在させず、このリポジトリでは`task dl-kaggle-comp`を使って`project.yml`の`data.raw_dir`へ同期する。

### Download Model
```python
path = kagglehub.model_download("owner/model/framework/variation")
```

### Execute a Kaggle Notebook
```bash
uv run kaggle kernels push -p ./notebook-dir
uv run kaggle kernels logs -f username/kernel-slug
uv run kaggle kernels output username/kernel-slug --path /tmp/kaggle-output/kernel-slug
```

Kaggle CLI 2.2.3 では notebook のログ取得を `kaggle kernels logs -f owner/slug` に統一する。`-f` は Kaggle UI と同系統の live SSE から stdout/stderr を逐次取得する。`--interval` は使わない。`kaggle kernels status`は診断用の補助情報に限り、完了判定のpollingには使わない。

See `.agents/skills/kaggle-platform/modules/kllm/scripts/cli_execute.sh` for a complete push-follow-download workflow.

### Competition data and submission

This repository is configured for a Notebook-only code competition. Use
`cli_competition.sh <competition> [download-dir]` only as a generic inspection
helper. Its default `data/raw/<competition>` destination does not update the
repository's configured `data.raw_dir`; use `task dl-kaggle-comp` for repository
data sync. The helper does not submit.
After validating the selected notebook output, submit exactly once with the
repository's `task submit-code` workflow. Do not use direct file upload here.

### Read Competition Discussions
```bash
uv run kaggle competitions topics list competition-name --sort-by recent --page-size 50 -v
uv run kaggle competitions topics show competition-name TOPIC_ID
```

For saved local notes, use `kaggle-discussion-archive`.

### Read Kaggle Forums
```bash
uv run kaggle forums list -v
uv run kaggle forums topics list product-announcements --sort-by recent --page-size 50 -v
uv run kaggle forums topics show product-announcements/TOPIC_ID
```

### Benchmarks CLI
```bash
uv run kaggle benchmarks init -y
uv run kaggle benchmarks tasks list
```

### Read Competition Overview Pages

Before joining or analyzing a competition, pull its overview pages (rules,
evaluation, data description, FAQ, prizes, timeline) via the
`list_competition_pages` MCP endpoint. This requires an API token; legacy
username/key credentials alone cannot authenticate the MCP call:

```bash
# Print every page as JSON
uv run python .agents/skills/kaggle-platform/modules/kllm/scripts/list_competition_pages.py --competition titanic

# One-line-per-page summary with key-page detection
uv run python .agents/skills/kaggle-platform/modules/kllm/scripts/list_competition_pages.py --competition titanic --summary

# Just the rules / evaluation page content
uv run python .agents/skills/kaggle-platform/modules/kllm/scripts/list_competition_pages.py --competition titanic --page rules
uv run python .agents/skills/kaggle-platform/modules/kllm/scripts/list_competition_pages.py --competition titanic --page evaluation
```

Works for regular competitions, playground series, AND hackathons. For
hackathon-specific overview content (judge ids, track structure), prefer
`hackathon_overview.py` from the hackathon module which calls
`get_hackathon_overview` instead.

See [references/competition-overview.md](references/competition-overview.md)
for the full endpoint reference, page-name conventions, and recommended
analysis patterns.

## Scripts

The paths below are relative to this `modules/kllm/` directory. Repository-root commands use the full `.agents/skills/kaggle-platform/modules/kllm/...` path.

- `../../shared/check_all_credentials.py` — Verify Kaggle credentials without printing their values
- `../registration/scripts/configure_token.py` — Store an API token entered locally with hidden input
- `scripts/repo_uv_env.sh` — Internal shared initialization for writable repo-local uv execution; do not invoke directly
- `scripts/network_check.sh` — Check network reachability to Kaggle API endpoints
- `scripts/poll_kernel.sh <kernel-slug> [output-dir]` — Legacy filename; follow live logs and then download output without status polling
- `scripts/cli_download.sh` — Download datasets and models via kaggle-cli
- `scripts/cli_execute.sh <notebook-dir> <kernel-slug> [output-dir]` — Execute a Kaggle Notebook
- `scripts/cli_competition.sh <competition> [download-dir]` — Inspect a competition and download raw competition data; never submit
- `scripts/cli_publish.sh <dataset|notebook|model> <dir> [model-handle]` — Publish resources
- `scripts/kagglehub_download.py` — Download datasets and models via kagglehub
- `scripts/kagglehub_publish.py <dataset|model> <handle> <local-dir> [version-notes]` — Publish via kagglehub
- `scripts/list_competition_pages.py --competition <slug> [--summary|--page NAME|--pretty]` — Fetch host-authored overview pages (rules, evaluation, data-description, FAQ, prizes, timeline) via the `list_competition_pages` MCP endpoint

## References

- [kaggle-knowledge.md](references/kaggle-knowledge.md) — Comprehensive Kaggle platform knowledge
- [kagglehub-reference.md](references/kagglehub-reference.md) — Full kagglehub Python API
- [cli-reference.md](references/cli-reference.md) — Complete kaggle-cli command reference
- [mcp-reference.md](references/mcp-reference.md) — Kaggle MCP server endpoint, auth, and tools
- [competition-overview.md](references/competition-overview.md) — `list_competition_pages` endpoint, page-name conventions, briefing patterns
