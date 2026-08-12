# exp069_pixiux_pf_beam_direct_submit_audit セッションノート

## 目的

exp063 の strict Pixiux public replay では LightGBM `lgb_mean` を提出して Public LB 8.811 だった。今回は LightGBM booster を通さず、Pixiux likelihood-PF / PF-Beam direct prediction の `likpf_mean` をそのまま提出し、exp063 と exp027 PF route anchor との差分を確認する。

## 現在の状態

- Route: pf_beam
- 状態: 実装済み、Kaggle inference 実行前
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
uv run python scripts/new_steering.py --experiment exp069_pixiux_pf_beam_direct_submit_audit
uv run python scripts/new_experiment.py --name exp069_pixiux_pf_beam_direct_submit_audit
uv run python -m py_compile experiments/exp069_pixiux_pf_beam_direct_submit_audit/public_notebook_replay_audit.py experiments/exp069_pixiux_pf_beam_direct_submit_audit/settings.py
uv run python -m json.tool experiments/exp069_pixiux_pf_beam_direct_submit_audit/exp069_pixiux_pf_beam_direct_submit_audit_train.ipynb
uv run python -m json.tool experiments/exp069_pixiux_pf_beam_direct_submit_audit/exp069_pixiux_pf_beam_direct_submit_audit_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp069_pixiux_pf_beam_direct_submit_audit
uv run ruff check experiments/exp069_pixiux_pf_beam_direct_submit_audit/public_notebook_replay_audit.py experiments/exp069_pixiux_pf_beam_direct_submit_audit/settings.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp069_pixiux_pf_beam_direct_submit_audit --notebook inference --kernel-id kentookumura/exp069-pixiux-pf-beam-direct-infer --title "exp069 pixiux pf beam direct infer" --run-on-push --strict
uv run python scripts/update_experiment_summary.py
```

- `py_compile`: PASS
- train notebook JSON validation: PASS
- inference notebook JSON validation: PASS
- `validate_experiment.py`: PASS
- `ruff check`: PASS
- Kaggle inference package generated at `experiments/exp069_pixiux_pf_beam_direct_submit_audit/kaggle/inference`
- Kaggle inference metadata at initial implementation: `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`
- Kaggle kernel sources: `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer`, `kentookumura/exp027-public-replay-needless090-sel15-spread3`
- `experiment_summary.md` updated with exp069 as `PF/Beam`, status `implemented`
- 2026-06-13: `kaggle kernels push -p experiments/exp069_pixiux_pf_beam_direct_submit_audit/kaggle/inference`
  - Kernel version 1 pushed: `kentookumura/exp069-pixiux-pf-beam-direct-infer`
  - Pull existence check succeeded at `/tmp/kaggle-pull/exp069-pixiux-pf-beam-direct-infer-v1`.
  - Initial normal logs were empty.
  - `logs -f` polling was interrupted by user question about GPU necessity.
  - exp069 does not need GPU because it bypasses LightGBM training/inference and only runs direct PF/Beam / likelihood-PF feature generation. `runtime.kaggle.enable_gpu` was changed to `false`; the corrected package should be pushed as version 2.
- 2026-06-13: CPU metadata package regenerated and pushed.
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp069_pixiux_pf_beam_direct_submit_audit --notebook inference --kernel-id kentookumura/exp069-pixiux-pf-beam-direct-infer --title "exp069 pixiux pf beam direct infer" --run-on-push --strict`
  - `kaggle kernels push -p experiments/exp069_pixiux_pf_beam_direct_submit_audit/kaggle/inference`
  - Kernel version 2 pushed successfully with `enable_gpu=false`.
  - Pull existence check succeeded at `/tmp/kaggle-pull/exp069-pixiux-pf-beam-direct-infer-v2`.
  - Initial normal logs and output were empty.
  - Short `logs -f` polling also returned no log before user confirmed Kaggle UI showed the notebook as running and requested no further monitoring.
- 2026-06-13: local polling process ended and returned final Kaggle logs.
  - Kernel version 2 completed.
  - CPU-only run generated `submission.csv` in about 147 sec wall log time.
  - Feature generation elapsed: 104.426 sec.
  - Total direct audit elapsed: 109.247 sec.
  - Output downloaded to `/tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v2`.
  - `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v2/submission.csv`: PASS.
  - `submission.csv`: 14,151 rows, columns `id,tvt`, SHA256 `c09a79b708a2a6e55696973412367943835437edb42e09668b80857099be4bf5`.
  - selected candidate: `likpf_mean`.
  - prediction range: 11600.931641 - 12241.389648; mean 11909.848657; std 280.543157.
  - fallback rows: 0.
  - reference diff vs exp027 submission: RMSE 11.930143, mean diff 6.218584, changed_rows 14,151, corr 0.999376.
  - reference diff vs exp063 submission: RMSE 7.724065, mean diff 4.319402, changed_rows 14,148, corr 0.999748.
  - Small audit artifacts were synced into `artifacts/`; `submission.csv` remains under Kaggle output path.
- 2026-06-13: user completed code submission.
  - Latest complete submission from `kaggle competitions submissions rogii-wellbore-geology-prediction -v`: `ref=53637978`, Public LB `9.877`.
  - A preceding pending row `ref=53637695` was visible in the submissions list, but the complete scored row is `53637978`.
  - Recorded `SUBMISSIONS.md` v029.
  - Result: worse than exp063 `8.811` by `+1.066` and worse than exp027 `8.781` by `+1.096`; do not adopt `likpf_mean` direct PF/Beam submission.
- 2026-06-15: reproducibility patch implemented.
  - Added `audit.deterministic: true` and set `runtime.num_workers: 1`.
  - Deterministic mode forces `CFG.n_jobs=1` to avoid `joblib.Parallel(... prefer="threads")` RNG interleaving.
  - Added stable per-well seed generation from `split:well_id`.
  - `_pf_ancc` and `_pf_z` now receive explicit seeds and call `np.random.seed(seed_base)` inside the numba kernel.
  - `lik_pf` is called with a stable per-well `seed_base`.
  - JIT warm-up no longer consumes unseeded Python-side `np.random.randn` arrays.
  - Stable seeds are bounded below the signed int32 edge so `seed_base + s` remains safely inside the expected NumPy seed range.
  - Rebuilt the Kaggle inference package with `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`, and `audit.deterministic: true`.
  - Validation after patch:
    - `uv run python -m py_compile experiments/exp069_pixiux_pf_beam_direct_submit_audit/public_notebook_replay_audit.py experiments/exp069_pixiux_pf_beam_direct_submit_audit/settings.py`: PASS
    - `uv run ruff check experiments/exp069_pixiux_pf_beam_direct_submit_audit/public_notebook_replay_audit.py experiments/exp069_pixiux_pf_beam_direct_submit_audit/settings.py`: PASS
    - `uv run python scripts/validate_experiment.py --experiment exp069_pixiux_pf_beam_direct_submit_audit`: PASS
    - `uv run python -m json.tool experiments/exp069_pixiux_pf_beam_direct_submit_audit/metrics.json`: PASS
    - `uv run python -m json.tool experiments/exp069_pixiux_pf_beam_direct_submit_audit/exp069_pixiux_pf_beam_direct_submit_audit_inference.ipynb`: PASS
  - The original v2 Public LB 9.877 belongs to the pre-patch implementation; a new Kaggle run is required to verify deterministic SHA / LB for the patched implementation.
- 2026-06-15: deterministic patch 版を Kaggle inference v3 として実行開始。
  - Pre-push validation:
    - `uv run python scripts/validate_experiment.py --experiment exp069_pixiux_pf_beam_direct_submit_audit`: PASS
    - `uv run python -m json.tool experiments/exp069_pixiux_pf_beam_direct_submit_audit/metrics.json`: PASS
    - `uv run ruff check experiments/exp069_pixiux_pf_beam_direct_submit_audit/public_notebook_replay_audit.py experiments/exp069_pixiux_pf_beam_direct_submit_audit/settings.py`: PASS
  - Rebuilt package:
    - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp069_pixiux_pf_beam_direct_submit_audit --notebook inference --kernel-id kentookumura/exp069-pixiux-pf-beam-direct-infer --title "exp069 pixiux pf beam direct infer" --run-on-push --strict`
    - `kernel-metadata.json`: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`.
    - packaged `config.yaml`: `audit.deterministic: true`.
  - Push:
    - `kaggle kernels push -p experiments/exp069_pixiux_pf_beam_direct_submit_audit/kaggle/inference`
    - Result: `Kernel version 3 successfully pushed`.
    - URL: `https://www.kaggle.com/code/kentookumura/exp069-pixiux-pf-beam-direct-infer`
  - Existence check:
    - `kaggle kernels pull kentookumura/exp069-pixiux-pf-beam-direct-infer -p /tmp/kaggle-pull/exp069-pixiux-pf-beam-direct-infer-v3 -m`: PASS.
  - Initial logs/output:
    - `kaggle kernels logs kentookumura/exp069-pixiux-pf-beam-direct-infer`: empty immediately after push.
    - `timeout 90 kaggle kernels logs -f --interval 10 kentookumura/exp069-pixiux-pf-beam-direct-infer`: no CLI log before timeout.
    - `kaggle kernels output kentookumura/exp069-pixiux-pf-beam-direct-infer -p /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3`: no files yet.
    - Treat as Kaggle API log/output lag after successful push and pull, not a slug mismatch.
- 2026-06-15: deterministic patch 版 Kaggle inference v3 completion confirmed.
  - `kaggle kernels logs kentookumura/exp069-pixiux-pf-beam-direct-infer`: completed log retrieved.
  - `kaggle kernels output kentookumura/exp069-pixiux-pf-beam-direct-infer -p /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3`: output downloaded.
  - Log evidence:
    - `Test dir: /kaggle/input/competitions/rogii-wellbore-geology-prediction/test wells= 3`
    - `public replay test wells: train_imputer_wells=773 test=3 | n_jobs=1 | PF seeds=128 | particles=500`
    - `building Pixiux/public base features from raw test files...`
    - `test base features: rows=14,151 cols=197 elapsed=38.2s`
    - `building Pixiux likelihood-PF replay features for test...`
    - summary: `deterministic=true`, `n_jobs=1`, `test_rows=14151`, `test_likpf_rows=14151`, `fallback_rows=0`.
  - `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3/submission.csv`: PASS.
  - v3 `submission.csv` SHA256: `57d5c55c5caa1d07b6691a054116b434d63dd9f8e03c73dfb6ef45753aa8fa01`.
  - v3 prediction range: 11601.626953 - 12241.427734; mean 11910.589483; std 280.495213.
  - v3 diff vs pre-patch v2 output: RMSE 1.120224, mean diff 0.740826, changed_rows 14,036, corr 0.999995528.
  - v3 reference diff vs exp027 submission: RMSE 12.852195, mean diff 6.959410, changed_rows 14,151, corr 0.999290.
  - v3 reference diff vs exp063 submission: RMSE 8.700711, mean diff 5.060228, changed_rows 14,151, corr 0.999689.
  - This confirms PF/Beam and likelihood-PF are regenerated from raw Kaggle test files during inference, not read from exp063 output. Reference submissions are used only for comparison artifacts.
- 2026-06-15: user completed deterministic v3 code submission.
  - Correct exp069 v3 complete submissions from `kaggle competitions submissions rogii-wellbore-geology-prediction -v`:
    - `ref=53706005`, date `2026-06-15 10:02:40.313000`, Public LB `9.721`.
    - `ref=53705994`, date `2026-06-15 10:02:19.190000`, Public LB `9.721`.
  - Recorded `ref=53706005` as the main v3 deterministic submission.
  - v3 deterministic `submission.csv` SHA256: `57d5c55c5caa1d07b6691a054116b434d63dd9f8e03c73dfb6ef45753aa8fa01`.
  - Result is worse than exp063 v2/best `8.811` by `+0.910` and worse than exp027 `8.781` by `+0.940`.
  - `ref=53710264` / `ref=53710105` Public LB `8.766` were initially misattributed to exp069; they are not exp069 v3 and are not recorded as this experiment's result.
  - Pre-patch v2 submission `ref=53637978` / Public LB `9.877` is retained as failed pre-deterministic result; do not mix it with v3.

### 予定

```bash
uv run python scripts/validate_experiment.py --experiment exp069_pixiux_pf_beam_direct_submit_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp069_pixiux_pf_beam_direct_submit_audit --notebook inference --kernel-id kentookumura/exp069-pixiux-pf-beam-direct-infer --title "exp069 pixiux pf beam direct infer" --run-on-push --strict
kaggle kernels push -p experiments/exp069_pixiux_pf_beam_direct_submit_audit/kaggle/inference
kaggle kernels pull kentookumura/exp069-pixiux-pf-beam-direct-infer -p /tmp/kaggle-pull/exp069-pixiux-pf-beam-direct-infer-v3 -m
kaggle kernels logs kentookumura/exp069-pixiux-pf-beam-direct-infer
kaggle kernels output kentookumura/exp069-pixiux-pf-beam-direct-infer -p /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3/submission.csv
```

## 変更点

- `public_notebook_replay_audit.py` を exp063 からコピーし、`run_pixiux_pf_beam_direct_submit_audit` を追加。
- inference notebook は `likpf_mean` direct submission を生成する構成へ変更。
- train notebook は no-op audit note にする。新しい学習は行わない。
- config route を `pf_beam` にし、Kaggle kernel sources に exp063 inference / exp027 replay を追加して reference diff を試みる。

## 次のアクション

1. exp069 `likpf_mean` direct PF/Beam は deterministic v3 でも Public LB 9.721 のため採用しない。
2. 今後は direct submission ではなく、PF/Beam disagreement、confidence、error map の診断値として使う。
