# exp198_exact_replacement_prune_on_exp148 セッションノート

## 目的

`exact_replacement_prune_on_exp148` を実装する。親は ML route submitted anchor の `exp148_learned_likelihood_fulltrain_addonly_on_exp092` とし、exp148 の active feature surface から高信頼の exact replacement / sign-flip / constant duplicate 17 列だけを削る。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle scoring 完了。Public LB 7.930、exp148 CPU runtime 7.921 には届かないため未採用
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 監査元: `studies/feature_replacement_audit/outputs/corr_prune_sanity_readout_on_exp148/`

## 実装メモ

- exp148 compact self-contained train / inference を元に、canonical notebook source を `exp198_exact_replacement_prune_on_exp148_train.py` / `exp198_exact_replacement_prune_on_exp148_inference.py` として作成した。
- `feature_columns_for_variant()` に `drop_columns` を追加し、削除対象が assembled features に存在しない場合は fail する。
- `expected_feature_count=277` を active variant に入れ、exp148 294 features から 17 列だけ削れたことを fail-fast で確認する。
- 新規特徴量、direct TVT replacement、blend、postprocess、submit は入れていない。

## Kaggle train push 前ガード

- active variants: 1
  - `drop_exact_replacements_17`
- disabled variants:
  - `exp148_historical_control`。保存済み exp148 CV / Public LB を comparison baseline とし、control は再学習しない。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- 合計 booster: 15
- control 再学習: なし
- parent / baseline 再学習: なし

## 削除対象 17 列

```text
sc_trust
ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt
ll_candidate_tvt_beam_mean_minus_last_known_tvt
ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt
ll_candidate_tvt_hyb_minus_last_known_tvt
ll_candidate_tvt_likpf_mean_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_last_known_tvt
ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt
ll_candidate_tvt_sc_ens_minus_last_known_tvt
tda0
dense_bias
uproj_beam_mean_resid
uproj_beam_med_resid
uproj_diff_pf_ancc_minus_pf_z
uproj_likpf_mean_resid
uproj_pf_ancc_resid
uproj_pf_z_resid
```

## 実行ログ

- 2026-07-05: `.steering/20260705-exp198-exact-replacement-prune-on-exp148/` を作成。
- 2026-07-05: `experiments/exp198_exact_replacement_prune_on_exp148/` を exp148 から作成。
- 2026-07-05: config、canonical train/inference source、記録ファイルを exp198 drop-only 実験として更新。
- 2026-07-05: 親コピー由来の古い exp148-named notebook source を削除し、canonical `exp198_exact_replacement_prune_on_exp148_train.py/ipynb` と `..._inference.py/ipynb` を正とした。
- 2026-07-05: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb` を train / inference に実行し、canonical `.ipynb` を生成。
- 2026-07-05: `jupytext --to ipynb --test` は train / inference とも pass。
- 2026-07-05: `.venv/bin/python -m py_compile` は train / inference とも pass。
- 2026-07-05: `.venv/bin/ruff check ... --select F821` は pass。
- 2026-07-05: `.venv/bin/python scripts/validate_experiment.py --experiment exp198_exact_replacement_prune_on_exp148` は strict pass。
- 2026-07-05: Kaggle train package を strict prepare 済み。
  - command: `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp198_exact_replacement_prune_on_exp148 --notebook train --kernel-id kentookumura/exp198-exact-replacement-prune-exp148-train --title 'exp198 exact replacement prune exp148 train' --run-on-push --strict`
  - metadata: GPU true、internet false、competition `rogii-wellbore-geology-prediction`、kernel sources `kentookumura/exp072-exp063-full-replay-feature-cache-train` / `kentookumura/exp145-train`
  - pushed package config SHA: loose package / notebook bootstrap `3f2972a11b64971923186aa301e011904d17a260e08550f8296008fadb457e5f`
  - note: push 後にローカル `config.yaml` の status を `running_kaggle_train_v1` に更新したため、現在の root config SHA は pushed package SHA とは異なる。
- 2026-07-05: `.venv/bin/python scripts/update_experiment_summary.py` を実行し、`experiment_summary.md` を 192 experiments で更新。
- 2026-07-05: Kaggle train v1 を push して実行開始。
  - command: `kaggle kernels push -p experiments/exp198_exact_replacement_prune_on_exp148/kaggle/train`
  - kernel: `kentookumura/exp198-exact-replacement-prune-exp148-train`
  - version: 1
  - URL: https://www.kaggle.com/code/kentookumura/exp198-exact-replacement-prune-exp148-train
  - push result: `Kernel version 1 successfully pushed`
  - initial status: `KernelWorkerStatus.RUNNING`
  - initial logs: empty. 実行中の Kaggle CLI logs が空になる既知挙動として扱い、失敗判定しない。
- 2026-07-05: Kaggle train v1 完了を確認。
  - final status: `KernelWorkerStatus.COMPLETE`
  - elapsed seconds: 12264.703
  - rows / wells / features: 3,783,989 / 773 / 277
  - feature join coverage: pass。dropped rows 0、dropped wells 0。
  - active variant: `drop_exact_replacements_17`
- 2026-07-05: Kaggle output を取得し、train artifact を確認。
  - command: `kaggle kernels output kentookumura/exp198-exact-replacement-prune-exp148-train -p /tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/train_v1`
  - output dir: `/tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/train_v1`
  - feature schema SHA256: `c9827f1a2fbec34e039035cab121b56077e56be8e1c3a74a7624ba205566c833`
  - model manifest SHA256: `f286ae46c6e47a66793ea2e4668e8569ef79fa6f896c19610c311ed1ff1c54d8`
  - OOF prediction gzip SHA256: `60e0756f6c137de676afac20686c5fa326214898cb4599f66d0d2f6690dc238e`
  - OOF prediction decompressed SHA256: `816dc0883b4920d7ece1ed63cc719dc11ae88dcb72d80a51ee2644076b41d381`
  - saved model count: 15
- 2026-07-05: pooled CV を exp148 GPU train baseline と比較。
  - `lgb0`: 8.525098952459498。exp148 同一 model 比 -0.07468690691939273。
  - `lgb1`: 8.531602620643975。exp148 同一 model 比 -0.03236850058569374。
  - `lgb2`: 8.476691203242892。exp148 同一 model 比 -0.03312851555118357。
  - `lgb_mean`: 8.457923652800986。exp148 `lgb_mean` 比 -0.043357529094834035。
- 2026-07-05: guard readout を確認。
  - 削除対象 17 列は feature schema に残っていない。
  - distance bucket: `000_050` -0.019642711、`050_100` -0.017495751、`1000_plus` -0.050682068 は改善。`100_250` +0.002599955、`250_500` +0.016456604、`500_1000` +0.010155678 は小幅悪化。
  - by-well: 423 wells 改善、350 wells 悪化。mean delta -0.025884613、median delta -0.017895937。
  - max regression: `b37fd114` +1.022149086 RMSE。max improvement: `86454a6f` -1.425857544 RMSE。
  - worst absolute wells: `1b1eba53` 47.337063、`86454a6f` 47.211758、`fb03ae90` 43.946739。
- 2026-07-05: `metrics.json`、`result.md`、`README.md` を completed train-side supported / no-submit として更新。
- 2026-07-05: inference port に進む。
  - selected variant: `drop_exact_replacements_17`
  - selected mode: `gpu_repro_guard_dp_threads8`
  - selected model: `lgb_mean`
  - train source: `kentookumura/exp198-exact-replacement-prune-exp148-train` version 1 output の saved boosters / manifest
  - expected model count: 15
  - expected feature count: 277
  - current-test learned likelihood features: exp145 rawtest artifact が sample ids と合わない場合は notebook 内で current test から再生成する。
  - Kaggle kernel sources: exp072 feature cache、exp198 train output、exp099、exp111、exp112。
  - 2026-07-05: `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb` を inference に実行し、canonical `.ipynb` を更新。
  - 2026-07-05: `jupytext --to ipynb --test`、`.venv/bin/python -m py_compile`、`.venv/bin/ruff check --select F821`、`.venv/bin/python scripts/validate_experiment.py --experiment exp198_exact_replacement_prune_on_exp148` は pass。
  - 2026-07-05: Kaggle inference package を strict prepare 済み。
    - command: `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp198_exact_replacement_prune_on_exp148 --notebook inference --kernel-id kentookumura/exp198-exact-replacement-prune-exp148-inference --title 'exp198 exact replacement prune exp148 inference' --run-on-push --strict`
    - metadata: GPU true、internet false、competition `rogii-wellbore-geology-prediction`
    - kernel sources: `kentookumura/exp072-exp063-full-replay-feature-cache-train` / `kentookumura/exp198-exact-replacement-prune-exp148-train` / `kentookumura/exp099-pf-multiobs-likelihood-train` / `kentookumura/exp111-learned-pf-likelihood-train` / `kentookumura/exp112-learned-pf-likelihood-followup-train`
    - bootstrap support files: 11
    - package config SHA: `e8854141e1940bcd2521ceb1ba318866cfd18ede4a48d8f4f4596df40f922363`
  - 2026-07-05: Kaggle inference v1 を push して実行開始。
    - command: `kaggle kernels push -p experiments/exp198_exact_replacement_prune_on_exp148/kaggle/inference`
    - kernel: `kentookumura/exp198-exact-replacement-prune-exp148-inference`
    - version: 1
    - URL: https://www.kaggle.com/code/kentookumura/exp198-exact-replacement-prune-exp148-inference
    - push result: `Kernel version 1 successfully pushed`
    - initial status: `KernelWorkerStatus.RUNNING`
    - initial logs: empty. 実行中の Kaggle CLI logs が空になる既知挙動として扱い、失敗判定しない。
  - 2026-07-05: ユーザーが Kaggle UI で inference v1 を停止。
    - observed log before stop: `building Pixiux likelihood-PF replay features for test...` 以降の出力なし。
    - diagnosis: model loading / LightGBM prediction ではなく、current-test likelihood-PF replay feature generation の `build_likpf(test_wids, "test")` 中。
    - exp148 / exp193 と同じ saved-booster inference path であり、LightGBM prediction 自体は CPU 実行。inference notebook metadata の GPU は不要。
  - 2026-07-05: inference v2 は CPU runtime へ変更。
    - config: `runtime.kaggle.inference.enable_gpu=false`
    - feature contract: unchanged。`PF_SEEDS=128`、`PF_PARTICLES=500`、selected variant / mode / model は v1 と同じ。
    - debug logging: likelihood-PF replay の well 単位 start/done/elapsed log を追加。
  - 2026-07-05: inference v2 CPU package を strict prepare 済み。
    - command: `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp198_exact_replacement_prune_on_exp148 --notebook inference --kernel-id kentookumura/exp198-exact-replacement-prune-exp148-inference --title 'exp198 exact replacement prune exp148 inference' --run-on-push --strict`
    - metadata: GPU false、internet false、competition `rogii-wellbore-geology-prediction`
    - kernel sources: `kentookumura/exp072-exp063-full-replay-feature-cache-train` / `kentookumura/exp198-exact-replacement-prune-exp148-train` / `kentookumura/exp099-pf-multiobs-likelihood-train` / `kentookumura/exp111-learned-pf-likelihood-train` / `kentookumura/exp112-learned-pf-likelihood-followup-train`
    - package config SHA: `edae7144fd06b5ae5590a4aa958a24d7c866a5f1bab311d296d6c1363e804b67`
  - 2026-07-05: Kaggle inference v2 CPU を push して実行開始。
    - command: `kaggle kernels push -p experiments/exp198_exact_replacement_prune_on_exp148/kaggle/inference`
    - kernel: `kentookumura/exp198-exact-replacement-prune-exp148-inference`
    - version: 2
    - URL: https://www.kaggle.com/code/kentookumura/exp198-exact-replacement-prune-exp148-inference
    - push result: `Kernel version 2 successfully pushed`
    - initial status: `KernelWorkerStatus.RUNNING`
    - initial logs: empty. 実行中の Kaggle CLI logs が空になる既知挙動として扱い、失敗判定しない。
  - 2026-07-05: v2 CPU push 後、数分後の再確認でも status は `KernelWorkerStatus.RUNNING`、CLI logs は empty。Kaggle UI 上の進捗確認待ち。
  - 2026-07-05: ユーザー報告で v2 CPU は `likpf test well 000d7d20: start` / `00bbac68: start` / `00e12e8b: start` から進まないことを確認。
    - diagnosis: `exp198_exact_replacement_prune_on_exp148_inference.py` の compact self-contained 化で親 exp148 self-contained inference の `@njit` デコレータが落ち、`_pf_lik_allseeds()` が pure Python 実行になっていた。
    - impact: LightGBM 推論や model loading ではなく、current-test likelihood-PF replay feature generation の 128-seed particle loop が極端に遅くなった。
    - fix: 親 exp148 と同じく `_interp1` / `_resamp` / `_beam_jit` / `_pf_ancc` / `_pf_z` / seeded PF helpers / `_pf_lik_allseeds` に Numba JIT を復元。`_pf_lik_allseeds` は `nogil=True` も復元。
  - 2026-07-05: inference v3 CPU / Numba 修正版の local validation を実施。
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `.venv/bin/python -m py_compile experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `.venv/bin/ruff check experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py --select F821`: pass
    - `.venv/bin/python scripts/validate_experiment.py --experiment exp198_exact_replacement_prune_on_exp148`: pass
  - 2026-07-05: Kaggle 側の v2 状態が `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` であることを確認。
  - 2026-07-05: inference v3 CPU / Numba 修正版 package を strict prepare 済み。
    - command: `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp198_exact_replacement_prune_on_exp148 --notebook inference --kernel-id kentookumura/exp198-exact-replacement-prune-exp148-inference --title 'exp198 exact replacement prune exp148 inference' --run-on-push --strict`
    - metadata: GPU false、internet false、competition `rogii-wellbore-geology-prediction`
    - generated notebook check: `@njit(cache=True)` / `@njit(cache=True, nogil=True)` present in package notebook.
  - 2026-07-05: Kaggle inference v3 CPU / Numba 修正版を push して実行開始。
    - command: `kaggle kernels push -p experiments/exp198_exact_replacement_prune_on_exp148/kaggle/inference`
    - kernel: `kentookumura/exp198-exact-replacement-prune-exp148-inference`
    - version: 3
    - URL: https://www.kaggle.com/code/kentookumura/exp198-exact-replacement-prune-exp148-inference
    - push result: `Kernel version 3 successfully pushed`
    - initial status: `KernelWorkerStatus.RUNNING`
    - immediate recheck: `KernelWorkerStatus.RUNNING`
    - immediate CLI logs: empty。実行中の Kaggle CLI logs が空になる既知挙動として扱い、失敗判定しない。
  - 2026-07-05: Kaggle inference v3 CPU / Numba 修正版は `KernelWorkerStatus.ERROR`。
    - v3 log: likelihood-PF replay は `000d7d20` 19.7s、`00e12e8b` 24.8s、`00bbac68` 30.1s で完了。Numba 修正は有効。
    - error: raw-test learned likelihood feature file が見つからず current-test 生成 fallback に入り、`candidate_specs_from_config()` で `TypeError: CandidateSpec() takes no arguments`。
    - diagnosis: compact self-contained 化で `CandidateSpec` の `@dataclass(frozen=True)` も落ちていた。親 self-contained inference には存在する。
    - fix: `CandidateSpec` と同種の `ExperimentPaths` に `@dataclass(frozen=True)` を復元。

  - 2026-07-05: inference v4 CPU / dataclass 修正版の local validation を実施。
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `.venv/bin/python -m py_compile experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py`: pass
    - `.venv/bin/ruff check experiments/exp198_exact_replacement_prune_on_exp148/exp198_exact_replacement_prune_on_exp148_inference.py --select F821`: pass
    - `.venv/bin/python scripts/validate_experiment.py --experiment exp198_exact_replacement_prune_on_exp148`: pass
  - 2026-07-05: inference v4 CPU / dataclass 修正版 package を strict prepare 済み。
    - command: `.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp198_exact_replacement_prune_on_exp148 --notebook inference --kernel-id kentookumura/exp198-exact-replacement-prune-exp148-inference --title 'exp198 exact replacement prune exp148 inference' --run-on-push --strict`
    - metadata: GPU false、internet false、competition `rogii-wellbore-geology-prediction`
    - generated notebook check: `@dataclass(frozen=True)` present for `ExperimentPaths` and `CandidateSpec`; `_pf_lik_allseeds` keeps `@njit(cache=True, nogil=True)`.
  - 2026-07-05: Kaggle inference v4 CPU / dataclass 修正版を push して実行開始。
    - command: `kaggle kernels push -p experiments/exp198_exact_replacement_prune_on_exp148/kaggle/inference`
    - kernel: `kentookumura/exp198-exact-replacement-prune-exp148-inference`
    - version: 4
    - URL: https://www.kaggle.com/code/kentookumura/exp198-exact-replacement-prune-exp148-inference
    - push result: `Kernel version 4 successfully pushed`
    - initial status: `KernelWorkerStatus.RUNNING`
    - immediate CLI logs: empty。実行中の Kaggle CLI logs が空になる既知挙動として扱い、失敗判定しない。
    - 120s recheck: `KernelWorkerStatus.RUNNING`。CLI logs は empty。v3 は約 93s で `CandidateSpec() takes no arguments` により ERROR になっていたため、少なくとも同じ即時失敗は再現していない。
  - 2026-07-05: Kaggle inference v4 完了を確認。
    - final status: `KernelWorkerStatus.COMPLETE`
    - notebook summary status: `inference_completed`
    - elapsed seconds: 155.668
    - feature replay elapsed seconds: 101.484
    - likelihood-PF test rows: 14,151 / 14,151
    - learned likelihood current-test features: 14,151 rows / 3 wells / 51 columns
    - selected variant / mode / model: `drop_exact_replacements_17` / `gpu_repro_guard_dp_threads8` / `lgb_mean`
    - model count: 15
    - feature count: 277
    - fallback rows: 0
    - prediction range / mean / std: 11590.4658203125 to 12240.234375 / 11905.457642403806 / 278.82843661424334
    - prediction SHA256: `e23bd8f8e59b56fe188833849075e1ce146ced28c2810ab0bd1ea0b42948944c`
    - submission SHA256: `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`
  - 2026-07-05: Kaggle output を取得し、submit-check を実施。
    - command: `kaggle kernels output kentookumura/exp198-exact-replacement-prune-exp148-inference -p /tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/inference_v4`
    - output dir: `/tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/inference_v4`
    - submission: `/tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/inference_v4/submission.csv`
    - check command: `.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp198_exact_replacement_prune_on_exp148/inference_v4/submission.csv --sample data/raw/sample_submission.csv`
    - submit-check result: FAIL なし、WARN なし、PASS。
    - rows / columns: 14,151 / 2
    - header: sample と一致 (`id`, `tvt`)
    - id order: sample と完全一致
    - duplicate ids: 0
    - missing / NaN / Inf-like values: 0
    - local submission SHA256: `e5b71f6f576a62567adfe189c2def12a7720375e264ce8c66b31456db7848c36`
  - 2026-07-05: scoring 完了を確認。
    - command: `kaggle competitions submissions rogii-wellbore-geology-prediction`
    - monitor command: `.venv/bin/python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp198_exact_replacement_prune_on_exp148 --competition rogii-wellbore-geology-prediction --once`
    - ref: `54354847`
    - submitted: `2026-07-05 07:51:41.043000`
    - status: `SubmissionStatus.COMPLETE`
    - public LB: `7.930`
    - private LB: not shown
    - attribution: latest completed scoring after exp198 inference v4 / submit-check; Kaggle description is blank.
    - comparison: exp148 GPU inference v7 Public LB 7.960 から -0.030 改善、exp193 Public LB 7.946 から -0.016 改善、exp148 CPU runtime Public LB 7.921 から +0.009 悪化。
    - decision: exp148 CPU runtime anchor は更新しない。exp198 は CV 改善が一部 LB に転移したが、現 ML route submitted anchor には届かない。

## 次アクション

exp198 は提出済みだが、exp148 CPU runtime anchor 7.921 を更新しない。次は exp198 の改善要因を維持しつつ CPU-runtime anchor との 0.009 差を埋められる低リスクな follow-up を検討する。
