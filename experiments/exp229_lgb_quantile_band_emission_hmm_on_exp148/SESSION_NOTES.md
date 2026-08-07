# exp229_lgb_quantile_band_emission_hmm_on_exp148 セッションノート

## 目的

`lgb_quantile_band_emission_hmm_on_exp148` backlog を実装する。exp221 の fixed sigma HMM を、LightGBM quantile band 由来の row-wise sigma で置き換え、LGB の予測信頼度を HMM emission に反映できるか train-side で確認する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train / HMM audit 完了、train-side 不採用
- CV: quantile-band HMM RMSE `8.684401098734789`
- LB: まだなし
- Notebook 構成:
  - `train`: q16/q50/q84 quantile LightGBM OOF と saved boosters
  - `train_aggregate`: quantile-band HMM train-side audit
  - `inference`: deferred
- 初回 train guard: 1 active variant x 1 LightGBM config x 3 quantiles x 5 folds = 15 boosters
- HMM audit: v2 の 3 variant は timeout。v3 は `lambda=0.25` の 1 variant、0 boosters で完走
- Parent/control retraining: なし

## 実装メモ

- `quantile_lgb_train.py`
  - exp148 と同じ feature surface を組み立て、`alpha=0.16/0.50/0.84` の quantile LightGBM を GroupKFold by well で学習する。
  - crossing 補正後の `q_low_tvt`, `q_mid_tvt`, `q_high_tvt`, `sigma_tvt` を `*_quantile_predictions.csv.gz` に保存する。
  - model manifest、feature importance、band summary、prediction SHA を保存する。
- `exact_hmm_smoother.py`
  - exp221 由来の HMM に row-wise `lgb_sigma` 対応を追加した。
  - `lgb_emission.sources.*.sigma_column` がある場合、ID aligned sigma Series を読み、floor/cap 後に emission に使う。
- `quantile_band_hmm_audit.py`
  - `run_train_feature_cache` で HMM cache を生成し、`run_direct_comparison` で exp221 相当の readout を出す。
- `config.yaml`
  - `experiment.route=ensemble`
  - train は GPU T4、train_aggregate は CPU override。
  - HMM lambda は `0.25/0.50/1.00`、sigma floor/cap は `6/30`。

## コマンドログ

```bash
python3 scripts/new_steering.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148
python3 scripts/new_experiment.py --name exp229_lgb_quantile_band_emission_hmm_on_exp148
```

- result: steering と実験 scaffold を作成。

## 予定コマンド

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train_aggregate.py
.venv/bin/python -m py_compile experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/*.py
.venv/bin/ruff check experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148 --select F821
python3 scripts/validate_experiment.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148
python3 scripts/prepare_kaggle_notebooks.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148 --notebook train --kernel-id kentookumura/exp229-lgb-quantile-band-exp148-train --title "exp229 lgb quantile band exp148 train" --run-on-push --strict
python3 scripts/prepare_kaggle_notebooks.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148 --notebook train_aggregate --kernel-id kentookumura/exp229-quantile-band-hmm-exp148-audit --title "exp229 quantile band hmm exp148 audit" --run-on-push --strict
```

## 2026-07-09 実装検証

```bash
.venv/bin/python -m py_compile experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train_aggregate.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/exp229_lgb_quantile_band_emission_hmm_on_exp148_train_aggregate.py
.venv/bin/ruff check experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148 --select F821
python3 scripts/validate_experiment.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148
python3 scripts/prepare_kaggle_notebooks.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148 --notebook train --kernel-id kentookumura/exp229-lgb-quantile-band-exp148-train --title "exp229 lgb quantile band exp148 train" --run-on-push --strict
python3 scripts/prepare_kaggle_notebooks.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148 --notebook train_aggregate --kernel-id kentookumura/exp229-quantile-band-hmm-exp148-audit --title "exp229 quantile band hmm exp148 audit" --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train/*.py experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train_aggregate/*.py
.venv/bin/ruff check experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train_aggregate --select F821
```

- result: すべて pass。
- `train` metadata: GPU true、internet false、machine_shape `NvidiaTeslaT4`、kernel id `kentookumura/exp229-lgb-quantile-band-exp148-train`。
- `train_aggregate` metadata: GPU false、internet false、machine_shape `None`、kernel id `kentookumura/exp229-quantile-band-hmm-exp148-audit`。
- `rg "__file__"`: notebook source 側には残存なし。
- `git status` はこの環境で `.git` を認識できず実行不可だったため、変更一覧は `find` / validation で確認した。

## 再現性メモ

- seed policy: GroupKFold は deterministic。LightGBM config seeds は exp063 family を継承。任意 subsampling は local `np.random.default_rng(42)`。
- stochastic components: LightGBM GPU 学習。
- CPU/GPU runtime: train は T4 GPU、`gpu_use_dp=true` / `deterministic=true` / `force_col_wise=true` / `num_threads=8`。HMM audit は CPU。
- HMM RNG: なし。outer parallel と Numba の浮動小数順序差は summary に記録する。
- Kaggle kernel id / version: 未実行。
- input / feature schema SHA: 未実行。
- feature content SHA: 未実行。
- model manifest / model SHA: 未実行。
- prediction SHA: 未実行。
- submission SHA: inference 未実装。
- rerun check: 未実行。

## 次のアクション

1. Jupytext 変換、py_compile、ruff F821、validate_experiment を通す。
2. Kaggle train push 前に package metadata と active booster 数を確認する。
3. train 完了後、train_aggregate HMM audit を実行して inference port の可否を判断する。

## 2026-07-11 Kaggle 実行

### train v1

```bash
kaggle kernels push -p experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels status kentookumura/exp229-lgb-quantile-band-exp148-train
kaggle kernels logs kentookumura/exp229-lgb-quantile-band-exp148-train
```

- result: version 1 は `ERROR`。
- failure: Kaggle source slug が古く、必要な親 kernel output が attach されなかったため、`exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` が見つからず停止。
- invalid sources reported by Kaggle: `kentookumura/exp072-cache`, `kentookumura/exp145-learned-ll-rawtest-parity`, `kentookumura/exp099-pf-multiobs-likelihood-probe`, `kentookumura/exp111-learned-pf-observation-likelihood-probe`, `kentookumura/exp112-learned-pf-likelihood-weight-or-feature`。

### source slug 修正

- `config.yaml` の `runtime.kaggle.train_kernel_sources` を以下へ修正。
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp145-train`
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp111-learned-pf-likelihood-train`
  - `kentookumura/exp112-learned-pf-likelihood-followup-train`
- `runtime.kaggle.train_aggregate_kernel_sources` の exp072 source も canonical slug へ修正。
- package 再生成、`validate_experiment`、package 側 `py_compile` は pass。

### train v2 予定

- active variants: 1 (`lgb_quantile_band_base`)
- LightGBM config count: 1
- quantile alpha count: 3 (`0.16/0.50/0.84`)
- folds: 5
- total booster count: 15
- parent/control retraining: なし
- push: `Kernel version 2 successfully pushed`。invalid source warning なし。
- initial status: `KernelWorkerStatus.RUNNING`。initial logs は空。
- follow-up status: 数分後も `KernelWorkerStatus.RUNNING`。logs はまだ空。train 完了後に `train_aggregate` を push する。
- result: version 2 は `ERROR`。
- failure: source attach は解決したが、`feature_groups` に存在しない `u_projection` を指定していたため `Unknown feature group ... u_projection` で停止。
- fix: `config.yaml` の active variant を exp148 と同じ `projection_correction` + `u_disagreement` + `learned_likelihood_confidence` に修正。
- validation after fix: `validate_experiment`、local/package `py_compile`、local/package `ruff --select F821` は pass。package 側 config も修正反映済み。
- train v3 push: `Kernel version 3 successfully pushed`。invalid source warning なし。
- train v3 status: version 2 の失敗時刻を越えても `KernelWorkerStatus.RUNNING`。logs はまだ空。train 完了待ち。
- train v3 follow-up: 追加監視後も `KernelWorkerStatus.RUNNING`。長時間 train として継続中。`train_aggregate` は train 完了後に push する。
- train v3 complete: `KernelWorkerStatus.COMPLETE`。
- train v3 result:
  - actual booster count: 15
  - rows / wells / features: 3,783,989 / 773 / 294
  - elapsed seconds: 10,374.504
  - q50 train-side CV RMSE: 8.684997693339485
  - raw q50 RMSE before crossing correction: 8.674055500815957
  - q16 pooled RMSE: 11.001985411166508
  - q84 pooled RMSE: 10.32602731757991
  - corrected band coverage: 0.5252380490535252
  - raw band coverage: 0.5229188034108979
  - band width mean / p50 / p90: 9.038330767901694 / 7.6328125 / 16.6044921875
  - crossing any / low-mid / mid-high: 0.03863383323788732 / 0.013713834791803041 / 0.024935590457583253
  - sigma effective mean / p50 / p90: 6.600344693730652 / 6.0 / 8.30224609375
  - sigma floor / cap rate: 0.7672646511393135 / 0.0
  - prediction gzip SHA: `4e93b79e4686d11d94e0af74110f421cc7652f20443c61cec8de84173945d9b1`
  - metrics SHA: `f14de515fed9a82e5f042fbef7299f570c4104b364da8e910c53c2ca48857eef`
  - model manifest SHA: `a7c0503b26258228a4b37350c8c2045f0e7f7ddc919068ca7253a0b01d349bbc`

### train_aggregate 予定

- active HMM variants: 3 (`lambda=0.25/0.50/1.00`, sigma floor/cap `6/30`)
- LightGBM booster count: 0
- parent/control retraining: なし
- Kaggle source: `kentookumura/exp229-lgb-quantile-band-exp148-train` の完了 output を参照する。
- train_aggregate v1 push: `Kernel version 1 successfully pushed` だが、`kentookumura/exp115-hidden-like-spatial-holdout-from-ppt` が invalid source warning。hidden-like readout 欠落を避けるため、`exp115-hidden-like-spatial-holdout-from-ppt-train` に修正して package 再生成。
- validation after exp115 source fix: `validate_experiment` と package `py_compile` は pass。metadata の kernel source と config の input path は `exp115-hidden-like-spatial-holdout-from-ppt-train` に修正済み。
- train_aggregate v2 push: `Kernel version 2 successfully pushed`。invalid source warning なし。
- train_aggregate v2 status: 数分後も `KernelWorkerStatus.RUNNING`。Kaggle CLI logs は実行中のため空。CPU HMM audit の完了待ち。

### train_aggregate v2 timeout recovery

- status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。Kaggle 12-hour 上限で停止し、最終 HMM cache / comparison artifact は未生成。
- progress: 773 well 中 654 start、652 complete。最後に complete した well は `d9eca87b`、last start は `[654/773]`。
- partial aggregation (652 completed wells):
  - `hmm_lgb_exp229_quantile_band_band_sf0600_sc3000_l0250`: mean well RMSE `6.316157991796836`、median `4.849422262757551`、best-well count `358`。
  - `hmm_lgb_exp229_quantile_band_band_sf0600_sc3000_l0500`: mean well RMSE `6.355612405646024`、median `4.833447313477735`、best-well count `91`。
  - `hmm_lgb_exp229_quantile_band_band_sf0600_sc3000_l1000`: mean well RMSE `6.385538783483597`、median `4.8947069119538105`、best-well count `203`。
- recovery decision: exp221 v2-to-v3 と同じく、partial score 最良の `lambda=0.25` だけを v3 で再実行する。`lambda=0.50/1.00` は後続の独立 audit に退避。
- v3 guard: 1 quantile source x 1 lambda x 1 floor/cap = 1 HMM variant、LightGBM booster 0、parent/control retraining なし。feature cache expected feature count は 17 から 7 に更新。

```bash
python3 scripts/validate_experiment.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148
python3 scripts/prepare_kaggle_notebooks.py --experiment exp229_lgb_quantile_band_emission_hmm_on_exp148 --notebook train_aggregate --kernel-id kentookumura/exp229-quantile-band-hmm-exp148-audit --title "exp229 quantile band hmm exp148 audit" --run-on-push --strict
kaggle kernels pull kentookumura/exp229-quantile-band-hmm-exp148-audit -p /tmp/kaggle-pull/exp229-quantile-band-hmm-exp148-audit -m
kaggle kernels push -p experiments/exp229_lgb_quantile_band_emission_hmm_on_exp148/kaggle/train_aggregate
kaggle kernels status kentookumura/exp229-quantile-band-hmm-exp148-audit
```

- v3 push: `Kernel version 3 successfully pushed`。同じ canonical kernel を使用し、invalid source warning なし。
- v3 initial status: `KernelWorkerStatus.RUNNING`。実行中の `kaggle kernels logs` は空であり、Kaggle CLI の通常挙動。

### train_aggregate v3 complete

```bash
kaggle kernels status kentookumura/exp229-quantile-band-hmm-exp148-audit
kaggle kernels logs kentookumura/exp229-quantile-band-hmm-exp148-audit
```

- status: `KernelWorkerStatus.COMPLETE`。
- audit elapsed: `11,320.273` sec、HMM generation `11,165.761` sec。
- HMM cache: 3,783,989 rows / 773 wells / 7 features、ok 773 / skipped 0。
- selected candidate: `hmm_lgb_exp229_quantile_band_band_sf0600_sc3000_l0250`。
- HMM metrics: RMSE `8.684401098734789`、MAE `5.088581187714929`、within10 `0.8526758402310366`。
- delta RMSE: exp148 `lgb_mean` 比 `+0.1831101144352214`、exp193 `lgb_mean` 比 `+0.227725045644398`。q50 `8.685006142759928` からは `-0.000605044025139` のみ改善。
- quantile train diagnostics: corrected band coverage `0.5252380490535252`、crossing any `0.03863383323788732`、sigma floor/cap `0.7672646511393135` / `0.0`。q50 と row-wise sigma は既存 point-prediction anchor を上回る emission surface にならなかった。
- exp221 fixed-sigma HMM (`8.32773695096078`) より `+0.356664147774009` 悪い。
- SHA: HMM feature decompressed `330d6fc3192b93c26a9ba022486fa5fe0a64d9bd923892f9454e6fad47c55a05`、gzip `6fb1e912f28fcd767e79d712f07296a086fe4d8bc5efb9872a88e1ac5d574777`、audit summary `2a620bd72784edbec84d5ec5ad47dd026242a4a33994cb9f36c14ac5c492551f`。
- decision: train-side 不採用。inference / submit / 未実行 `lambda=0.50/1.00` の追加 audit は行わない。Kaggle logs に最終 metrics / artifact 名 / SHA が出ており、後続入力や提出物は不要なため output archive は取得しない。
