# exp222_row_step_delta_target_ablation_on_exp148 セッションノート

## 目的

exp148 learned-likelihood feature surface の target-only ablation。教師を anchor residual から row-to-row step delta に変え、OOF 評価では predicted step delta を well ごとに累積して TVT に戻す。ユーザー指定により、まず CPU の `lgb0` だけで反証する。

## 現在の状態

- Route: ml_model
- 状態: running_on_kaggle_v1
- CV: まだなし
- LB: なし
- 実行中: Kaggle CPU train `kentookumura/exp222-stepdelta-lgb0` v1

## 実行計画

- enabled variant: 1 (`step_delta_target_lgb0`)
- active mode: 1 (`cpu_deterministic_threads8`)
- LightGBM config: 1 (`lgb0`, `lgb_config_indices: [0]`)
- folds: 5
- 合計 booster 数: 5
- 親 exp148 control 再学習: なし
- GPU 使用: なし

## コマンドログ

```bash
make new-steering EXP=exp222_row_step_delta_target_ablation_on_exp148
make new-exp EXP=exp222_row_step_delta_target_ablation_on_exp148
.venv/bin/python -m py_compile experiments/exp222_row_step_delta_target_ablation_on_exp148/row_step_delta_target_ablation_on_exp148.py
.venv/bin/python -m py_compile experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
.venv/bin/ruff check experiments/exp222_row_step_delta_target_ablation_on_exp148/row_step_delta_target_ablation_on_exp148.py experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py --select F821
make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148
rg -n "^# %% \\[markdown\\]|^# #|^# ##|^# [0-9]\\." experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_train.py experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
wc -l experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_train.py experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py experiments/exp222_row_step_delta_target_ablation_on_exp148/row_step_delta_target_ablation_on_exp148.py
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-row-step-delta-target-ablation-on-exp148-train --title 'exp222 row step delta target ablation on exp148 train' --run-on-push --strict"
cat experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train/kernel-metadata.json
rg -n "enable_gpu|lgb_config_indices|step_delta|kernel_sources|exp222_row_step_delta|row_step_delta_target" experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train
kaggle kernels push -p experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-row-step-delta-target-train --title 'exp222 row step delta target train' --run-on-push --strict"
kaggle kernels push -p experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train
kaggle kernels list --mine --page-size 20 --sort-by dateRun
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb0
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb1
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb2
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-inference
kaggle kernels status kentookumura/exp206-dz-dtvt-bpeak-cluster-train
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-row-step-delta-target-train --title 'exp222 row step delta target train' --run-on-push --strict"
make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148
kaggle kernels pull kentookumura/exp222-row-step-delta-target-train -p /tmp/kaggle-pull/exp222-row-step-delta-target-train -m
kaggle kernels list --mine --page-size 20 --sort-by dateRun
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb0
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb1
kaggle kernels status kentookumura/exp220-row-neighbor-exp148-lgb2
kaggle kernels status kentookumura/exp209-joint-exact-parity-train
kaggle kernels push -p experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train
kaggle kernels pull kentookumura/exp072-exp063-full-replay-feature-cache-train -p /tmp/kaggle-pull/exp072-exp063-full-replay-feature-cache-train -m
kaggle kernels pull kentookumura/exp145-train -p /tmp/kaggle-pull/exp145-train -m
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-stepdelta-lgb0 --title 'exp222 stepdelta lgb0' --run-on-push --strict"
kaggle kernels push -p experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train
kaggle kernels status kentookumura/exp222-stepdelta-lgb0
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-stepdelta-lgb0 --title 'exp222 stepdelta lgb0' --run-on-push --strict"
make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148
make update-summary
kaggle kernels status kentookumura/exp222-stepdelta-lgb0
kaggle kernels logs kentookumura/exp222-stepdelta-lgb0
```

### 検証結果

- `py_compile`: helper / train source とも pass。
- Jupytext: `.py -> .ipynb` 変換 pass、`--test` pass。
- `ruff --select F821`: pass。
- `make validate-exp`: strict pass。
- `__file__` check: train source / inference notebook に該当なし。
- notebook 構成: parent compact は 1702 行、exp222 は train notebook source 214 行 + helper 1959 行。exp222 notebook は setup、input/target contract、lgb0 train、metrics/生成物表示の 4 章構成で、薄い `main()` 呼び出しのみではない。
- Kaggle package: `experiments/exp222_row_step_delta_target_ablation_on_exp148/kaggle/train` を作成。metadata は `enable_gpu=false`、`enable_internet=false`、`run_on_push=true`、competition source `rogii-wellbore-geology-prediction`、kernel sources `kentookumura/exp072-exp063-full-replay-feature-cache-train` / `kentookumura/exp145-train`。
- Bootstrap support manifest: `config.yaml`、train source、helper、settings、project.yml、src を含む。package 側 config で `lgb_config_indices: [0]` と `step_delta_target` を確認済み。
- Initial push failed with Kaggle `SaveKernel` 400 and no detailed message for kernel id `kentookumura/exp222-row-step-delta-target-ablation-on-exp148-train`. id/title slug matched, so the next retry will keep the same experiment but shorten the canonical slug/title to `kentookumura/exp222-row-step-delta-target-train` / `exp222 row step delta target train`.
- Shortened package metadata was regenerated successfully with `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`, competition source, and the same exp072/exp145 kernel sources.
- Shortened push failed to start execution with Kaggle message: `Maximum batch CPU session count of 5 reached.`
- Running sessions checked:
  - `kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train`: RUNNING
  - `kentookumura/exp220-row-neighbor-exp148-lgb0`: RUNNING
  - `kentookumura/exp220-row-neighbor-exp148-lgb1`: RUNNING
  - `kentookumura/exp220-row-neighbor-exp148-lgb2`: RUNNING
  - `kentookumura/exp209-joint-exact-parity-train`: RUNNING
- Complete sessions checked: `exp206-dz-dtvt-bpeak-cluster-inference`, `exp206-dz-dtvt-bpeak-cluster-train`.
- Do not stop existing running kernels without user approval. Retry exp222 push after one CPU session completes, or stop a user-approved existing session first.
- After recording the blocker in `config.yaml` / `metrics.json`, the short-slug Kaggle package was regenerated. Package config now also has `status: implemented_push_blocked_cpu_session_limit`.
- Final `make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148`: strict pass.
- Short-slug kernel existence check via `kaggle kernels pull kentookumura/exp222-row-step-delta-target-train -m` returned Kaggle `GetKernel` 500, so existence remains inconclusive. Treat the next action as retrying the same short-slug push after CPU slot frees.
- 2026-07-08 retry:
  - CPU slots had freed: `exp220-row-neighbor-exp148-lgb0` and `exp220-row-neighbor-exp148-lgb1` were COMPLETE; `exp221`, `exp220 lgb2`, and `exp209` remained RUNNING.
  - Retry push to `kentookumura/exp222-row-step-delta-target-train` returned `Kernel push error: Notebook not found`.
  - Source kernels `kentookumura/exp072-exp063-full-replay-feature-cache-train` and `kentookumura/exp145-train` were both pullable, so source-not-found was ruled out.
  - To avoid the broken short slug, the same exp was re-prepared as `kentookumura/exp222-stepdelta-lgb0` / `exp222 stepdelta lgb0`.
  - Push succeeded: `Kernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/kentookumura/exp222-stepdelta-lgb0`.
  - Status check: `kentookumura/exp222-stepdelta-lgb0` is `KernelWorkerStatus.RUNNING`.
  - Repo config/metrics/README/result were updated to `running_on_kaggle_v1`; package was regenerated after the status-only config update for future pushes. The already-running v1 used the same training logic.
  - Final local checks: `make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148` pass; `make update-summary` updated `experiment_summary.md`.
- Follow-up status check remained RUNNING. `kaggle kernels logs kentookumura/exp222-stepdelta-lgb0` returned empty output, which is expected for running notebooks in this CLI environment.
- 2026-07-08 v1 failure investigation:
  - `kaggle kernels status kentookumura/exp222-stepdelta-lgb0`: `KernelWorkerStatus.ERROR`.
  - `kaggle kernels logs kentookumura/exp222-stepdelta-lgb0`: `DeadKernelError: Kernel died` at about 498 seconds.
  - Last normal notebook output was `learned feature rows: 3783989 wells: 773 columns: 51`; no LightGBM variant/fold output appeared.
  - Root cause assessment: the input contract cell used `load_learned_likelihood_ml_features()` for preview, loading and converting the full exp145 learned likelihood cache before the main training path loaded it again. This created an unnecessary Kaggle CPU RAM peak before lgb0 training.
  - Fix for v2: train notebook preview now reads only header plus `nrows=8`; helper uses ordered-key fast paths to avoid large learned-feature `merge` copies where possible, and drops temporary DataFrames with `gc.collect()`.
  - Checks after fix: `py_compile` pass, `ruff --select F821` pass, Jupytext `.py -> .ipynb` pass, Jupytext `--test` pass, `make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148` pass.
  - Package regenerated for same kernel id `kentookumura/exp222-stepdelta-lgb0` with CPU / run_on_push / exp072+exp145 sources.
  - Push succeeded: `Kernel version 2 successfully pushed. Please check progress at https://www.kaggle.com/code/kentookumura/exp222-stepdelta-lgb0`.
  - Status check: `kentookumura/exp222-stepdelta-lgb0` is `KernelWorkerStatus.RUNNING`.
  - Immediate logs were empty, which is expected for running notebooks in this CLI environment.
  - After a 5 minute wait, status became `KernelWorkerStatus.ERROR`.
  - v2 logs: `learned feature preview rows: 8 columns: 51` then `DeadKernelError: Kernel died` at about 353 seconds. No LightGBM variant/fold output appeared.
  - Root cause assessment update: preview full-load was fixed, but full feature assembly still exceeded Kaggle CPU memory before LightGBM started.
  - Fix for v3: replace full-frame finite checks with column-wise checks; replace anchor merge with well-key mapping; rewrite step-delta target creation to sort only a small order frame instead of copying/sorting the full feature frame; drop feature columns after matrix creation for single variant/mode while retaining `last_known_tvt` and `md_since`; delete fold-level model/prediction temporaries; add stage logs with peak RSS.
  - Checks after v3 fix: `py_compile` pass, `ruff --select F821` pass, Jupytext `--test` pass, `make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148` pass.
  - Package regenerated for same kernel id `kentookumura/exp222-stepdelta-lgb0`.
  - Push succeeded: `Kernel version 3 successfully pushed. Please check progress at https://www.kaggle.com/code/kentookumura/exp222-stepdelta-lgb0`.
  - Status check: `kentookumura/exp222-stepdelta-lgb0` is `KernelWorkerStatus.RUNNING`.
  - Immediate logs were empty, which is expected for running notebooks in this CLI environment.
- 2026-07-09 v3 completion:
  - User reported completion; `kaggle kernels status kentookumura/exp222-stepdelta-lgb0` returned `KernelWorkerStatus.COMPLETE`.
  - `kaggle kernels logs kentookumura/exp222-stepdelta-lgb0` confirmed `train_completed`.
  - Output was downloaded to `/tmp/exp222-stepdelta-lgb0-v3` because bucket / by-well / cumulative drift tables were needed for train-side rejection evidence.
  - Runtime stage logs:
    - loaded exp072 cache: 3,783,989 rows / 199 columns / peak RSS 17,665.0 MB.
    - attached learned features: 3,783,989 rows / 327 columns / peak RSS 21,972.9 MB.
    - built feature matrix: shape `[3783989, 294]`; feature columns were then dropped from the frame, leaving 41 columns.
  - Fold RMSE:
    - fold0: 14.576648195503386, best_iteration 131.
    - fold1: 15.961224816380408, best_iteration 116.
    - fold2: 14.807865232931501, best_iteration 143.
    - fold3: 16.35812490733958, best_iteration 137.
    - fold4: 14.71681228061527, best_iteration 129.
  - Pooled `lgb0` / `lgb_mean`: reconstructed TVT RMSE 15.301575123885728, step-delta RMSE 0.036165870680591676. `lgb_mean` is identical to `lgb0` because only lgb0 was active.
  - Baseline comparison:
    - exp148 lgb0 RMSE 8.59978585937889 -> delta +6.7017892645068375.
    - exp148 lgb_mean RMSE 8.50128118189582 -> delta +6.800293941989908.
  - Distance bucket vs exp148 lgb_mean:
    - 000_050: 0.6014456748962402 vs 0.978726, delta -0.37728032510375975.
    - 050_100: 1.3953514099121094 vs 1.316981, delta +0.07837040991210942.
    - 100_250: 2.541387796401977 vs 2.084639, delta +0.45674879640197696.
    - 250_500: 4.16950798034668 vs 3.298294, delta +0.8712139803466799.
    - 500_1000: 6.876345634460449 vs 4.792035, delta +2.084310634460449.
    - 1000_plus: 16.93307113647461 vs 9.325405, delta +7.607666136474609.
  - Worst wells: `1b1eba53` RMSE 67.72745513916016, `896d15b9` 58.371578216552734, `81bf5923` 51.91246795654297, `42c538a1` 51.65792465209961, `fb03ae90` 46.28654098510742.
  - Cumulative drift: top final errors include `1b1eba53` -69.5078125, `896d15b9` -84.615234375, `81bf5923` +78.115234375.
  - SHA evidence:
    - prediction lgb_mean SHA256: `5c807ad31c3ae3604a6c59b2bd130405b62c704b8a41ad6c364e80c7ba1e281e`.
    - model manifest SHA256: `a1bb1bbf20c35ffa23ecb8a4811d6314cd2b6e49c8ff57857f4b8181e4974d25`.
    - metrics SHA256: `4550862dc3cb3b67f54ea68d2371cae2b8f6d775b913e6b39eb7cb4ff6c661b0`.
    - bucket metrics SHA256: `d6e4bf7333716f04708ce966874184f3a888fc78bb590d487d1b58aa023a67cb`.
    - by-well SHA256: `eabd670bc2959a2a248287b8bc407872efa8f365b3ab6f582a968aff8dbeafca`.
    - cumulative drift SHA256: `945e594f30a4823f8e00de25d01f2420b645616d37772b437816bda795385eea`.
  - Decision: train-side rejected. Do not expand to lgb1/lgb2, do not port inference, and do not submit.

### Kaggle 実行予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py
.venv/bin/ruff check experiments/exp222_row_step_delta_target_ablation_on_exp148/row_step_delta_target_ablation_on_exp148.py experiments/exp222_row_step_delta_target_ablation_on_exp148/exp222_row_step_delta_target_ablation_on_exp148_train.py --select F821
make validate-exp EXP=exp222_row_step_delta_target_ablation_on_exp148
make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-row-step-delta-target-ablation-on-exp148-train --title 'exp222 row step delta target ablation on exp148 train' --run-on-push --strict"
make push-kaggle-train EXP=exp222_row_step_delta_target_ablation_on_exp148
```

## 変更点

- `row_step_delta_target_ablation_on_exp148.py`
  - exp148 helper を元に、`target_anchor_delta` / `target_tvt` / `target_step_delta` / `row_number` / `row_within_tail` を追加。
  - OOF prediction を `last_known_tvt + cumsum(pred_step_delta)` で TVT に復元。
  - metrics に `rmse_step_delta` と `rmse_anchor_delta` を追加。
  - `cumulative_drift.csv` を保存。
  - inference は未対応として `NotImplementedError` にした。
- `exp222_row_step_delta_target_ablation_on_exp148_train.py`
  - Kaggle CPU train notebook source。
  - 入力 preview、target contract、lgb0 training、metrics / bucket / by-well / cumulative drift / importance 表示を含む。
- `config.yaml`
  - CPU only、`lgb_config_indices: [0]`、5 folds、control 再学習なしを明記。

## 再現性メモ

- seed policy: LightGBM config seed と GroupKFold seed 42 を固定。
- stochastic components: LightGBM histogram / bagging。custom sampling は `max_train_rows: null` なので本実行では使わない。
- CPU/GPU runtime: CPU only、`deterministic: true`、`force_col_wise: true`、`n_jobs: 8`、`num_threads: 8`。
- Kaggle kernel id / version: `kentookumura/exp222-stepdelta-lgb0` version 3 complete。
- input / feature schema SHA: summary / manifest に保存済み。
- feature content SHA: exp072 source SHA256 `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`。
- model manifest / model SHA: manifest SHA256 `a1bb1bbf20c35ffa23ecb8a4811d6314cd2b6e49c8ff57857f4b8181e4974d25`、fold model SHA は `metrics.json` に記録済み。
- prediction SHA: lgb_mean OOF prediction SHA256 `5c807ad31c3ae3604a6c59b2bd130405b62c704b8a41ad6c364e80c7ba1e281e`。
- submission SHA: 生成しない。
- rerun check: 未実施。

## 次のアクション

1. 完了。不採用。
2. lgb1/lgb2 展開、inference port、submit は行わない。
3. recursive delta prediction は直接予測ではなく drift diagnostics / posthoc guard の材料に限定する。
