# exp217_grcal_public_raw_pf_confidence_features_on_exp158 セッションノート

## 2026-07-07 実装

`KAGGLE_DIRECTION.md` の backlog `grcal_public_raw_pf_confidence_features_on_exp158` を実装する。

### 狙い

exp214 の public-like raw PF diagnostics では `pf_raw_scale_12` が scoped train-side 64 wells で `pf_raw_scale_5` より良く、scale / seed / ESS / GR residual scale に selector confidence feature としての余地があった。一方、direct PF replacement や scoped output join は避けるべきなので、exp217 では full train の exp158 selector surface 上で同じ target-free public raw PF diagnostics を再生成し、候補別 confidence feature としてだけ使う。

### 実装内容

- `docs/legacy/steering/20260707-exp217-grcal-public-raw-pf-confidence-features-on-exp158/` を作成。
- `experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/` を exp184 から作成し、exp217 用にリネーム。
- exp214 の `public_raw_gr_residual_scale_control.py` を補助 helper として同梱。
- `grcal_public_raw_pf_confidence_features_on_exp158.py` に `add_public_raw_pf_features()` を追加。
  - exp214 scoped output は join しない。
  - raw horizontal GR、raw typewell GR、known prefix TVT_input/Z/MD、evaluation-row GR/Z/MD だけから full train features を再生成。
  - `pubraw_pf_scale5`, `pubraw_pf_scale12`, `pubraw_scale_spread`, `pubraw_gr_sigma`, `pubraw_ess_mean`, `pubraw_resampling_rate`, `pubraw_seed_weight_max`, `pubraw_seed_weight_entropy`, seed mean / best seed delta を生成。
  - row feature と candidate-long feature の両方に candidate distance / family interaction を追加。
- selectable candidates は exp158 と同じ 8 候補で固定。
- LightGBM は 3 configs x 5 folds = 15 boosters、parent/control retraining なし。
- exp158 と同じ Viterbi grid を評価。

### 実行前ガード

- active selector variant: 1
- LightGBM configs: 3
- folds: 5
- planned boosters: 15
- public raw PF: 500 particles x 128 seeds / well、scales `[3, 5, 8, 12]`
- full train target wells: `max_target_wells: null`
- GPU: disabled
- internet: disabled
- control / parent retraining: なし
- inference / submit: なし

### 再現性メモ

- public raw PF seed は exp214 helper と同じ stable SHA256 per query well / seed index。
- PF の stochastic components は particle propagation / resampling のみ。
- 並列 RNG は使わない。
- gzip 生成物は decompressed SHA を主証拠にする。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 静的検証

- `.venv/bin/python -m py_compile` / `python3 -m py_compile`: pass
- `.venv/bin/ruff check --select F821 experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158`: pass
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: train / inference pass
- `make validate-exp EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`: strict pass

### Kaggle 実行予定

- kernel id: `kentookumura/exp217-grcal-public-raw-pf-confidence-features-on-exp158-train`
- title: `exp217 grcal public raw pf confidence features on exp158 train`
- run-on-push: true
- active selector variant: 1
- LightGBM configs: 3
- folds: 5
- planned boosters: 15
- GPU: disabled
- internet: disabled
- control / parent retraining: なし

### 残作業

- Kaggle train 完了確認
- Kaggle output 取得

## 2026-07-07 Kaggle train push

### push attempt 1

- command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp217-grcal-public-raw-pf-confidence-features-on-exp158-train --title 'exp217 grcal public raw pf confidence features on exp158 train' --run-on-push --strict"`
- metadata: private, CPU, internet off, run-on-push true
- command: `make push-kaggle-train EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`
- result: `SaveKernel` 400 Bad Request
- follow-up: `kaggle kernels list --mine --search exp217` returned `Not found`; long slug/title likely exceeded Kaggle-side constraints. Repackage same exp with shorter id/title instead of creating a new experiment.

### push attempt 2

- command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train --title 'exp217 grcal pubraw pf conf exp158 train' --run-on-push --strict"`
- metadata: private, CPU, internet off, run-on-push true
- command: `make validate-exp EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`
- validation: strict pass
- command: `make push-kaggle-train EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`
- result: Kernel version 1 successfully pushed
- URL: https://www.kaggle.com/code/kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train
- Kaggle metadata pull: success, `id_no=126243481`, `machine_shape=None`
- `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`: `KernelWorkerStatus.RUNNING`
- initial `kaggle kernels logs`: empty; this is consistent with the known CLI behavior while a notebook is still running.

## 2026-07-08 Kaggle train v1 completion check

- user reported the run completed, so Kaggle status/logs were checked.
- `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
- `kaggle kernels logs kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` was saved to `/tmp/exp217_kaggle_logs.json` and parsed after removing the CLI version warning prefix.
- Log progress:
  - support files restored and config printed normally.
  - planned cost was 1 selector variant, 3 LightGBM configs, 5 folds, 15 boosters, no control retraining.
  - public raw PF generation progressed through 773 train wells; last periodic line was `[pubraw] 750/773`, and fold 0 started afterward, so feature generation reached the training stage.
  - fold 0 started at log time 39,726.537 sec.
  - fold 1 started at log time 41,247.829 sec.
  - fold 2 started at log time 42,594.767 sec; last log line was fold 2 training start at 42,641.658 sec.
- `kaggle kernels output` downloaded only partial output to `/tmp/kaggle-output/exp217_grcal_public_raw_pf_confidence_features_on_exp158/train`.
  - Saved model files exist for fold 0 and fold 1 only: 6 LightGBM text files.
  - No final summary, metrics CSV, OOF predictions, feature importance, or Viterbi outputs were present.
- Decision: v1 is `kaggle_train_v1_cancelled_no_cv`; it is not a valid CV run and must not be used for model selection.
- Runtime implication: the combined pubraw generation plus 15-booster selector training is too long for a single notebook run. A rerun needs either a cached `pubraw_` feature-generation stage or split train notebooks by fold/config after caching.

## 2026-07-11 Kaggle train v2 GPU switch

- User requested switching exp217 Kaggle execution to GPU.
- Push guard before GPU rerun:
  - active selector variant: 1
  - LightGBM configs: 3
  - folds: 5
  - planned boosters: 15
  - control / parent retraining: なし
- Runtime metadata: `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, internet off.
- Scope: same exp217 kernel slug and same model/feature config as v1; no selectable candidate, Viterbi grid, fold, or LightGBM parameter change.
- Caveat: `pubraw_` generation is CPU/Numba code, so the T4 runtime does not make that stage GPU-native. This switch only changes the Kaggle hardware/runtime for v2.
- Planned push target: `kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` version 2.

### v2 push and GPU verification follow-up

- command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train --title 'exp217 grcal pubraw pf conf exp158 train' --run-on-push --strict"`
- local metadata before push: `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, internet off, run-on-push true.
- command: `kaggle kernels push -p experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/kaggle/train --accelerator NvidiaTeslaT4`
- result: Kernel version 2 successfully pushed.
- `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`: `KernelWorkerStatus.RUNNING`
- `kaggle kernels list --mine --search exp217`: `lastRunTime` is `2026-07-10 15:28:26.043000` UTC, matching the v2 push time.
- `kaggle kernels logs` returned only the CLI version warning and no notebook output yet.
- Metadata caveat: `kaggle kernels pull -m` returned `enable_gpu: false`, `machine_shape: "None"`, which appears to be the previous materialized version metadata while v2 is still running. The local upload metadata and CLI `--accelerator NvidiaTeslaT4` request are the evidence for the T4 request.
- Follow-up: add an early `nvidia-smi` runtime guard and repush as version 3 so the notebook itself verifies GPU availability before entering the long CPU/Numba pubraw stage.

### v3 guard attempt

- A local early `nvidia-smi` guard was tested with Jupytext conversion, py_compile, and `make validate-exp`.
- command: `kaggle kernels push -p experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/kaggle/train --accelerator NvidiaTeslaT4`
- result: `Kernel push error: Maximum batch GPU session count of 2 reached.`
- Interpretation: the v3 guard version was not pushed. The error is consistent with Kaggle treating currently running batch notebooks as GPU sessions; v2 remains the active exp217 GPU run.
- After the failed v3 push, `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` remained `KernelWorkerStatus.RUNNING`.
- `kaggle kernels logs` still returned no notebook output after the CLI version warning.
- Local source was restored to the v2-equivalent notebook body so the repository state reflects the active pushed run; the session note retains the v3 guard attempt for traceability.

## 2026-07-13 Pubraw cache split

- User requested splitting `pubraw_` generation out of the main selector train and caching it.
- Latest Kaggle check for `kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`:
  - status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`
  - v2 log confirms `GPU enabled: True`, but only reached `[pubraw] 1/773`; no CV output was generated.
- Implementation change:
  - Added `run_public_raw_pf_cache_generation(...)` to generate `id + pubraw_*` cache artifacts without LightGBM training.
  - Added cache-read path to `add_public_raw_pf_features(...)`; downstream selector train now expects `exp217_grcal_public_raw_pf_confidence_features_on_exp158_pubraw_features.csv.gz`.
  - Added `exp217_grcal_public_raw_pf_confidence_features_on_exp158_pfbeam_features.py/ipynb` as the cache-generation notebook.
  - Updated `scripts/prepare_kaggle_notebooks.py` so `pfbeam_features` keeps its historical no-source default, but respects kind-specific configured sources when provided.
- Cache stage push guard:
  - active selector variants: 0
  - LightGBM configs: 0
  - folds: 0
  - planned boosters: 0
  - control / parent retraining: なし
  - public raw PF: 500 particles x 128 seeds / well, scales `[3, 5, 8, 12]`, full train wells
  - GPU: disabled, because pubraw generation is CPU/Numba
  - input kernel sources: exp099 train cache and exp072 full replay cache
- Downstream selector train plan after cache completion:
  - active selector variant: 1
  - LightGBM configs: 3
  - folds: 5
  - planned boosters: 15
  - pubraw generation: skipped; read from `kentookumura/exp217-pubraw-cache`
- Validation before cache push:
  - `.venv/bin/python -m py_compile` for helper, train notebook script, cache notebook script, and prepare script: pass
  - `.venv/bin/ruff check --select F821 experiments/exp217... scripts/prepare_kaggle_notebooks.py`: pass
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ..._pfbeam_features.py`: pass
  - `make validate-exp EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`: strict pass
- Cache package:
  - command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook pfbeam_features --kernel-id kentookumura/exp217-pubraw-cache --title 'exp217 pubraw cache' --run-on-push --strict"`
  - metadata: private, CPU, internet off, run-on-push true, competition source present, kernel sources `exp099` and `exp072`.
- Cache push attempt:
  - command: `kaggle kernels push -p experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/kaggle/pfbeam_features`
  - result: `Kernel push error: Maximum batch CPU session count of 5 reached.`
  - `kentookumura/exp217-pubraw-cache` is not created yet; `kaggle kernels status` returned 404.
  - Recent running CPU sessions include `rogii-exp239-pseudotail-augmentation-train`, `exp238-nested-rank-slot-exp218-train`, `exp242-two-regime-rate-noise-pf-train`, and exp241 containment shards 0/2/3. These were not stopped because they are separate experiments.
- Downstream train package check:
  - command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train --title 'exp217 grcal pubraw pf conf exp158 train' --run-on-push --strict"`
  - metadata: CPU, internet off, kernel sources `exp099`, `exp072`, `exp115`, and intended `exp217-pubraw-cache`.
  - train package now loads pubraw cache and no longer regenerates `pubraw_` inside the selector run.

## 2026-07-13 Pubraw cache Kaggle run

- User requested Kaggle execution for the split cache stage.
- Reprepared the `pfbeam_features` package and retried the original slug:
  - command: `kaggle kernels push -p experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/kaggle/pfbeam_features`
  - result: `Kernel push error: Notebook not found`
  - `kentookumura/exp217-pubraw-cache` still did not exist.
- Repackaged the same cache notebook with a fresh slug/title:
  - command: `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook pfbeam_features --kernel-id kentookumura/exp217-pubraw-cache-v1 --title 'exp217 pubraw cache v1' --run-on-push --strict"`
  - command: `kaggle kernels push -p experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/kaggle/pfbeam_features`
  - result: Kernel version 1 successfully pushed.
  - URL: https://www.kaggle.com/code/kentookumura/exp217-pubraw-cache-v1
  - `kaggle kernels status kentookumura/exp217-pubraw-cache-v1`: `KernelWorkerStatus.RUNNING`
  - initial `kaggle kernels logs` returned no notebook output after the CLI version warning; this is expected immediately after notebook start.
- Local config and downstream train source were updated to read cache artifacts from `kentookumura/exp217-pubraw-cache-v1`.

## 2026-07-14 Pubraw cache completion and selector train push guard

- User reported the cache run completed; Kaggle state was verified.
- `kaggle kernels status kentookumura/exp217-pubraw-cache-v1`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp217-pubraw-cache-v1` showed:
  - final status: `pubraw_cache_completed`
  - rows: 3,783,989
  - wells: 773
  - generated `pubraw_` feature columns: 25
  - runtime seconds: 23,098.604
  - output artifact: `exp217_grcal_public_raw_pf_confidence_features_on_exp158_pubraw_features.csv.gz`
  - `pubraw_features` sha256: `63ea14c78f980f1c18060923585797da86e73d3700a270afd340ea3a8be2381d`
  - `pubraw_features` decompressed sha256: `1c7c8717740d696bf28be7bf78e8fac9f33957957886dafff86c438a7e030e7d`
- `kaggle kernels files kentookumura/exp217-pubraw-cache-v1` listed the expected cache outputs, including `pubraw_features.csv.gz`, schema, feature summary, and summary JSON.
- Selector train push guard after cache completion:
  - active selector variants: 1
  - LightGBM configs: 3
  - folds: 5
  - planned boosters: 15
  - control / parent retraining: なし
  - GPU: disabled
  - pubraw generation inside train: skipped; reads `kentookumura/exp217-pubraw-cache-v1`
  - kernel sources: exp099, exp072, exp115, exp217-pubraw-cache-v1
- Validation before selector train push:
  - `python3 -m json.tool experiments/exp217_grcal_public_raw_pf_confidence_features_on_exp158/metrics.json`: pass
  - `make validate-exp EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`: strict pass
  - `make prepare-kaggle-notebooks EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train --title 'exp217 grcal pubraw pf conf exp158 train' --run-on-push --strict"`: pass
- Selector train v3 push:
  - command: `make push-kaggle-train EXP=exp217_grcal_public_raw_pf_confidence_features_on_exp158`
  - result: Kernel version 3 successfully pushed.
  - URL: https://www.kaggle.com/code/kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train
  - `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`: `KernelWorkerStatus.RUNNING`
  - initial `kaggle kernels logs` returned no notebook output after the CLI version warning; this is expected for a running Kaggle notebook in this environment.

## 2026-07-14 Selector train v3 completion

- User reported train v3 completed; Kaggle state was verified.
- `kaggle kernels status kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` showed final `status: completed_train_side_audit`.
- `kaggle kernels files kentookumura/exp217-grcal-pubraw-pf-conf-exp158-train` listed expected train-side artifacts: metrics, OOF predictions, by-well, bucket metrics, feature importance, feature schema, model manifest, pubraw schema/summary, score summary, subgroup metrics, viterbi params, and model files.
- `kaggle kernels output ... -p /tmp/kaggle-output/exp217_grcal_public_raw_pf_confidence_features_on_exp158/train_v3` was started to inspect metrics. Download was interrupted after the necessary small files were present locally; retrieved files include metrics, by-well, bucket metrics, feature importance, feature schema, and model manifest. The partial zero-byte OOF file was not used.
- Best result:
  - variant: `viterbi_sw050_bias000_jw050_jf025_d0075_std999999_md0000_seg012`
  - RMSE: `10.669620823863928`
  - MAE: `6.4532320890091945`
  - within 10ft: `0.7885488039209416`
  - oracle label accuracy: `0.2841279401182192`
  - path switches: `14,599` (`3.85809789616196` / 1000 rows)
- Comparisons:
  - vs exp158 continuity `10.789163253079206`: `-0.11954242921527758`
  - vs exp184 heatmap add-only `10.560650324533297`: `+0.1089704993306313`
  - vs exp191 typewell continuity `10.598006879875323`: `+0.07161394398860566`
  - vs best OOF `lgb_candidate_error_ranker` `10.6956862053139`: `-0.02606538144997117`
- Readout:
  - `1000_plus` distance bucket RMSE: `11.693821`
  - worst wells: `86454a6f` RMSE `57.240032`, `1b1eba53` RMSE `56.474652`, `5f4d2a52` RMSE `44.241381`
  - `pubraw_` features were high-importance: `pubraw_gr_sigma` rank 4 in error ranker and rank 3 in multiclass.
- Decision: train-side positive vs exp158, but not a PF/Beam route anchor update. Do not port to inference or submit from exp217.

## 2026-07-14 Closeout

- User requested closing the experiment.
- Final status: `closed_train_side_positive_vs_exp158_not_anchor_no_submit`.
- `kaggle-review-exp` reviewer was run for `exp217`; core evidence categories are present across steering, experiment docs, `metrics.json`, `result.md`, `experiment_summary.md`, and `KAGGLE_DIRECTION.md`.
- `KAGGLE_DIRECTION.md` status was changed from `完了・保留` to `完了・不採用`.
- No inference notebook execution, submission generation, or additional Kaggle run is planned for exp217.
- Follow-up, if the signal is reused later, should be a new scoped hypothesis: add-only `pubraw_` confidence features on a stronger surface such as exp184/exp191+, or a limited high-spread/high-likpf-gap guard. Do not revive exp217 as a direct replacement/blend/submission path.
