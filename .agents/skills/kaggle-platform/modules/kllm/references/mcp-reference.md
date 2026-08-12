# Kaggle MCP Server Reference

> Official docs: https://www.kaggle.com/docs/mcp
> Blog: https://www.kaggle.com/blog/kaggles-official-mcp-server

## Endpoint

```
https://www.kaggle.com/mcp
```

Protocol: Streamable HTTP (MCP standard).

## Authentication

Pass your Kaggle API token as a Bearer token:

```
Authorization: Bearer <your_api_token>
```

Use the API token from "Generate New Token" at [kaggle.com/settings](https://www.kaggle.com/settings). Credentialの生成と保存は[registrationの認証設定](../../registration/references/kaggle-setup.md)を正とし、このreferenceへ複製しない。

## Client Configuration

Configure the endpoint in the MCP client, but keep the token in the client's
local secret store or a protected environment-variable reference. Do not paste
the token into a shell command, process argument, committed JSON, terminal
output, or an agent conversation. If a client only supports literal headers,
enter the token locally in an untracked, permission-restricted configuration.

## Repository helper

Repository scripts use `shared/mcp_client.py`, which reads locally configured
credentials and sends the Authorization header in-process. It does not invoke
`curl` with the token in process arguments. Bundled scripts are the preferred
entry points. For a one-off Python call executed from the repository root, use
this canonical import setup:

```python
import sys
from pathlib import Path

skill_root = Path(".agents/skills/kaggle-platform").resolve()
sys.path.insert(0, str(skill_root))

from shared.mcp_client import mcp_call, mcp_list_tools, resolve_token

token = resolve_token()
tools = mcp_list_tools(token=token)
```

## Tool Inventory (66 tools observed 2026-04-22, retested 2026-05-04)

Source: `tools/list` against `https://www.kaggle.com/mcp`, cross-referenced
against [shepsci/kmcp-tools](https://github.com/shepsci/kmcp-tools)
`data/endpoints.md`. These are dated observations, not current guarantees. Use
`tools/list` and a task-relevant probe to confirm against the server at runtime.

**Status flag changes since the 2026-04-22 audit** (observed in the upstream
live audit on 2026-05-04; the test itself is not included in this module):

- `get_hackathon_write_up` — was KNOWN_FAIL; returned PASS in the 2026-05-04 retest.
- `get_benchmark_leaderboard` — was BLOCKED (permission-gated); returned PASS
  with the API token used in the 2026-05-04 retest. This does not define a token class.
- `get_competition` for classic competitions (titanic, playground-series-s6e2)
  — was KNOWN_FAIL; returned PASS in the 2026-05-04 retest.

The hackathon module's `fetch_writeup.py` fallback chain (`get_writeup` →
`get_writeup_by_topic` → `get_writeup_by_slug`) is retained as defensive
plumbing because live endpoint behavior can change.

Status legend:
- ✅ verified PASS (as of 2026-05-04)
- 🔒 BLOCKED by role/permission (host or judge required)
- 🔬 BAD_PROBE (test infra issue, tool may still work)

### Auth
- ✅ `authorize` — Check whether the client can authorize with Kaggle
- ✅ `get_user_profile` — Fetch a public user profile

### Competition
- ✅ `get_competition` — Backend bug for classic competitions (titanic, playground-series-s6e2) was fixed between 2026-04-22 and 2026-05-04
- ✅ `search_competitions`
- ✅ `get_competition_data_files_summary`
- ✅ `get_competition_leaderboard`
- ✅ `get_competition_submission`
- ✅ `search_competition_submissions`
- ✅ `list_competition_data_files`
- ✅ `list_competition_data_tree_files`
- ✅ `list_competition_pages` — host-authored overview pages (rules, evaluation, data-description, FAQ, prizes, timeline). Universal: works for regular competitions, playgrounds, and hackathons. See [competition-overview.md](competition-overview.md) for the full reference and patterns. Wrapper script: `modules/kllm/scripts/list_competition_pages.py`.
- ✅ `download_competition_data_file`
- ✅ `download_competition_data_files`
- ✅ `download_competition_leaderboard`
- ✅ `start_competition_submission_upload`
- ✅ `submit_to_competition`
- 🔒 `create_code_competition_submission` — kernel→competition; permission-gated

### Dataset
- ✅ `search_datasets`
- ✅ `get_dataset_info`
- ✅ `get_dataset_metadata`
- ✅ `get_dataset_status`
- ✅ `get_dataset_files_summary`
- ✅ `list_dataset_files`
- ✅ `list_dataset_tree_files`
- ✅ `download_dataset`
- ✅ `update_dataset_metadata`
- ✅ `upload_dataset_file`

### Notebook
- ✅ `search_notebooks`
- ✅ `get_notebook_info`
- ✅ `get_notebook_session_status`
- ✅ `create_notebook_session`
- ✅ `cancel_notebook_session`
- ✅ `download_notebook_output`
- ✅ `download_notebook_output_zip`
- ✅ `list_notebook_files`
- ✅ `list_notebook_session_output`
- ✅ `save_notebook`

### Model
- ✅ `list_models`
- ✅ `get_model`
- ✅ `create_model`
- ✅ `update_model`
- ✅ `list_model_variations`
- ✅ `get_model_variation`
- ✅ `update_model_variation`
- ✅ `list_model_variation_versions`
- ✅ `list_model_variation_version_files`
- ✅ `download_model_variation_version`

### Forum
- ✅ `list_forums`
- ✅ `list_forum_topics`
- ✅ `get_forum`
- ✅ `get_forum_topic`

### Hackathon (newer surface — see `modules/kllm/hackathon/`)
- ✅ `get_hackathon_overview` — rules, eligibility, rubric, prizes
- ✅ `list_hackathon_write_ups` — submission roster (paginated)
- ✅ `list_hackathon_tracks` — resolve track id → title
- ✅ `get_hackathon_write_up` — generic invocation error fixed between 2026-04-22 and 2026-05-04; `get_writeup` remains the simpler interface (no `competitionName` arg needed)
- ⚠️  `download_hackathon_write_ups` — host-only; may return CSV header only

### Writeup
- ✅ `get_writeup` — preferred full-body fetch (use over `get_hackathon_write_up`)
- ✅ `get_writeup_by_slug`
- ✅ `get_writeup_by_topic`
- ⚠️  `get_resolved_writeup_links` — host context returns `{}`; participants get role-gated denial

### Benchmark
- ✅ `create_benchmark_task_from_prompt`
- ✅ `get_benchmark_leaderboard` — permission gate differed between the 2026-04-22 and 2026-05-04 audits; recheck with a task-relevant probe

### Episode (simulation/agent evaluation)
- 🔬 `get_episode_agent_logs`
- 🔬 `get_episode_replay`
- ✅ `list_submission_episodes`

### Search
- ✅ `search_content` — generic content search

## Usage Patterns

### Search and Download
```
Search datasets matching "titanic" → select best match → download it
```

### Competition Workflow
```
List competitions → join → download data → submit predictions → check leaderboard
```

### Publish Resources
```
Create private dataset with title and license → upload files → verify
```

### Execute Notebook
```
Push notebook code → poll status → retrieve output when complete
```

### Hackathon Writeup Retrieval
```
get_hackathon_overview (rules/rubric) → list_hackathon_tracks (id→title) →
list_hackathon_write_ups (roster) → get_writeup per submission →
get_resolved_writeup_links (host/judge only)
```

`get_hackathon_write_up` failed in the 2026-04-22 audit and returned PASS in
the 2026-05-04 retest. Treat both results as dated evidence and probe the live
endpoint before depending on it. The
`modules/kllm/hackathon/scripts/fetch_writeup.py` script starts with
`get_writeup` because it has the simpler argument shape, then falls back to
`get_writeup_by_topic` and `get_writeup_by_slug`.

## Official Documentation

- Full tool reference: https://www.kaggle.com/docs/mcp
- Blog announcement: https://www.kaggle.com/blog/kaggles-official-mcp-server
- MCP Protocol spec: https://modelcontextprotocol.io
