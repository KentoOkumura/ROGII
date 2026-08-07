# exp221_lgb_oof_gaussian_emission_hmm_on_exp148 セッションノート

## 目的

`lgb_oof_gaussian_emission_hmm_on_exp148` backlog を実装する。exp148 / exp193 の LightGBM OOF 点予測を exp209 exact HMM の Gaussian observation likelihood として追加し、HMM を ML 予測込みの sequence smoother として反証する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train v3 complete。train-side positive、inference / submit は未実施
- CV / LB: train-side OOF RMSE 8.32773695096078 / LB なし
- 初回 active variants: exp148 `lgb_mean` x `sigma=[8, 12, 20]` x `lambda=0.50` = 3 variants
- v3 active variants: exp148 `lgb_mean` x `sigma=[20]` x `lambda=0.50` = 1 variant
- GPU cost: なし。CPU-only HMM generation / readout
- LightGBM config / folds / boosters: 0 / 0 / 0
- Parent/control retraining: なし
- Inference / submit: なし
- Kaggle train v1: `kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train` version 1, status ERROR。notebook top-level `kernelspec` 欠落で Papermill 起動前に失敗。
- Kaggle train v2: same kernel, version 2, status `CANCEL_ACKNOWLEDGED` / timeout near 12h before final artifacts
- Kaggle train v3: same kernel, version 3, status COMPLETE with single `sigma=20` variant
- Kaggle URL: https://www.kaggle.com/code/kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train

## 実装メモ

- `exact_hmm_smoother.py`
  - exp209 の exact HMM に optional `lgb_tvt`, `lgb_sigma`, `lgb_lambda` を追加。
  - `logP_lgb = -0.5 * ((state_tvt - pred_tvt_lgb) / sigma)^2` を GR emission に足す。
  - LGB OOF prediction source を読み、variant ごとに wide columns を出力する。
- `direct_hmm_comparison.py`
  - exp072 baseline、exp148/exp193 LGB OOF baseline、HMM+LGB candidates を同じ readout で比較。
  - overall、distance bucket、exp115 hidden-like subgroup、by-well、HMM std calibration、step-delta を保存。
- `joint_cache_generation.py`
  - exp072 再生成は default disabled。
  - HMM cache generation と direct comparison を orchestrate。
- train notebook
  - setup、cost guard、input/OFF source contract、HMM run、metrics/artifacts 保存をセル分割。

## 固定パラメータ

- HMM grid: `step=0.35`, `n_rates=41`, `band_pad=100.0`
- HMM transition / GR emission: exp209 / exp205 default
- HMM speed setting: `feature_cache.hmm.outer_workers=2`, `runtime.numba_num_threads=2`
- LGB emission initial grid: `sigma=[8.0, 12.0, 20.0]`, `lambda=[0.50]`
- Deferred full grid: `sigma=[8, 12, 20]`, `lambda=[0.25, 0.50, 1.00]`

## コマンドログ

```bash
python3 scripts/new_steering.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148
python3 scripts/new_experiment.py --name exp221_lgb_oof_gaussian_emission_hmm_on_exp148 --source experiments/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation
```

- result: steering と実験 scaffold を作成。

```bash
python3 scripts/validate_experiment.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148
.venv/bin/python -m py_compile experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train/exp221_lgb_oof_gaussian_emission_hmm_on_exp148_train.py
.venv/bin/ruff check experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train --select F821
kaggle kernels push -p experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train
kaggle kernels pull kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -p /tmp/kaggle-pull/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -m
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels logs kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
```

- result: strict validation pass、package-side py_compile pass、ruff F821 pass。Kaggle kernel version 1 pushed。
- push 前 guard: active variants 3、LightGBM config / folds / boosters 0 / 0 / 0、GPU false、parent/control retraining なし、inference/submit なし。
- Kaggle metadata: `id_no=126348332`, `enable_gpu=false`, `enable_internet=false`, `machine_shape=None`。
- 2026-07-08 19:28 JST status: `KernelWorkerStatus.RUNNING`。CLI logs は実行中のため空。
- 2026-07-08 19:30 JST status: `KernelWorkerStatus.ERROR`。logs: `ValueError: No kernel name found in notebook and no override provided.`

```bash
python3 scripts/prepare_kaggle_notebooks.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148 --notebook train --kernel-id kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train --title "exp221 lgb oof gaussian emission hmm exp148 train" --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train/exp221_lgb_oof_gaussian_emission_hmm_on_exp148_train.py scripts/prepare_kaggle_notebooks.py
.venv/bin/ruff check experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train scripts/prepare_kaggle_notebooks.py --select F821
kaggle kernels push -p experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train
kaggle kernels pull kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -p /tmp/kaggle-pull/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -m
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels logs kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
```

- fix: `scripts/prepare_kaggle_notebooks.py` が package notebook の top-level `kernelspec` / `language_info` を補完するように変更。exp221 train / inference `.ipynb` にも同 metadata を追加。
- result: package notebook metadata に `kernelspec.name=python3` が入ったことを確認し、Kaggle kernel version 2 pushed。
- 2026-07-08 19:32 JST status: `KernelWorkerStatus.RUNNING`。CLI logs は実行中のため空。
- 2026-07-09: user reported timeout. CLI status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`。
- v2 progress from logs: 578 start lines, 576 completed well summaries, last start `[578/773]`, last completed well `c2c4db09` at 43165.445 sec. No final HMM feature cache / comparison artifacts were written because the implementation writes the full cache at the end.
- v2 partial best variant counts over 576 completed wells: `s2000/l0500` 306, `s0800/l0500` 167, `s1200/l0500` 103.
- v2 partial mean RMSE over 576 completed wells: `s2000/l0500` 6.182348, `s1200/l0500` 6.283716, `s0800/l0500` 6.345489.
- decision: v3 runs only `hmm_lgb_exp148_lgb_mean_s2000_l0500` (`sigma=20`, `lambda=0.50`) so expected runtime is roughly one third of v2 and should fit under the Kaggle limit.

```bash
python3 scripts/validate_experiment.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148
python3 scripts/prepare_kaggle_notebooks.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148 --notebook train --kernel-id kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train --title "exp221 lgb oof gaussian emission hmm exp148 train" --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train/exp221_lgb_oof_gaussian_emission_hmm_on_exp148_train.py scripts/prepare_kaggle_notebooks.py
.venv/bin/ruff check experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train scripts/prepare_kaggle_notebooks.py --select F821
kaggle kernels push -p experiments/exp221_lgb_oof_gaussian_emission_hmm_on_exp148/kaggle/train
kaggle kernels pull kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -p /tmp/kaggle-pull/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -m
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels logs kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
```

- v3 push guard: active variants 1 (`sigma=20/lambda=0.50`), LightGBM config / folds / boosters 0 / 0 / 0、GPU false、parent/control retraining なし、inference/submit なし。
- v3 package validation: strict validation pass、package-side py_compile pass、ruff F821 pass、package config has `max_variants=1`, `expected_feature_count=7`, top-level `kernelspec.name=python3`。
- v3 Kaggle metadata: `id_no=126348332`, `enable_gpu=false`, `enable_internet=false`, `machine_shape=None`。
- 2026-07-09 12:17 JST status: `KernelWorkerStatus.RUNNING`。CLI logs は起動直後のため空。

```bash
kaggle kernels status kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train
kaggle kernels logs kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train > /tmp/exp221_kaggle_logs_v3.json
kaggle kernels output kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-train -p /tmp/kaggle-output/exp221_train_v3
```

- 2026-07-09 status: `KernelWorkerStatus.COMPLETE`。
- v3 runtime: `elapsed_seconds=17827.454`、HMM generation elapsed 17,565.669 sec、773 wells、3,783,989 rows。
- selected candidate: `hmm_lgb_exp148_lgb_mean_s2000_l0500` (`sigma=20.0`, `lambda=0.50`, exp148 `lgb_mean`)。
- overall RMSE: 8.32773695096078、MAE 4.811969896535815、within10 0.8588101075346678。
- delta RMSE: vs exp148 `lgb_mean` -0.17355403333878705、vs exp193 `lgb_mean` -0.12893910212961046、vs exp072 `likpf_mean` -3.267160717480211。
- distance buckets: `000_050`, `050_100`, `100_250`, `250_500`, `500_1000`, `1000_plus` のすべてで exp148 / exp193 比改善。`1000_plus` は exp148 比 -0.1941791770714811、exp193 比 -0.1426262529964859。
- exp115 hidden-like: `verification_like_spatial` RMSE 9.57222044736198、exp148 比 -0.2296047139454629、exp193 比 -0.1140876587982617。`verification_like_typewell_purged` RMSE 9.545366072857329、exp148 比 -0.2318095495900944、exp193 比 -0.1150297621715541。
- by-well: exp148 比 509 wells 改善 / 264 wells 悪化、最大悪化 `2e63d9de` +4.981191458319891 RMSE。exp193 比 495 wells 改善 / 278 wells 悪化、最大悪化 `2e63d9de` +5.628247835746583 RMSE。
- step-delta: HMM+LGB `abs_step_delta_mean=0.0110347859097897`, p99 0.0650000000005093, `>5/10/25` rates 0。
- HMM std calibration: lowest std bin RMSE 8.985650889209955、highest std bin RMSE 9.99710877081125。middle bins は低いが単調 calibration ではないため、posterior std を calibrated confidence として直接使わない。
- SHA: HMM feature decompressed content `ceca23fbd6b2f85a4e2d7e351f6922de41dd244f8eff2a03c282aed742dcd2b8`、gzip `8027ac8840d1048cfbda8377bcbae6a9b47b50ad7765da38eebb7f1df57d0a54`、joint summary `01dbed0fb52b1a6a8b12606bdc43c5e0b3d2c313f27872d506dfeee90d55e9c1`。
- generated artifacts: overall, distance bucket, hidden-like, by-well, HMM std calibration, step-delta, feature schema, by-well generation summary, HMM train feature cache, comparison summary, joint summary。

## 次のアクション

1. 同じ exp221 内で inference port に進むか判断する。
2. 進める場合は `sigma=20/lambda=0.50` のみを raw-test-safe に生成し、提出前に runtime、feature SHA、row count、fallback 0、sample submission 互換性を確認する。
3. 追加 grid は v2 timeout を踏まえ、別 run に分ける場合だけ検討する。

## 2026-07-09 inference 実装

- ユーザー指示により同じ exp221 内で inference port に進む。
- 方針: 見えている test は sample なので score checklist には使わない。Kaggle hidden test 実行時に exp148 saved LightGBM boosters で current-test `lgb_mean` 予測を生成し、その予測を exp221 HMM の Gaussian emission center として `sigma=20/lambda=0.50` だけ実行する。
- 実装: exp148 saved-model inference helper を exp221 package に同梱し、`exact_hmm_smoother.py` に test-row 用 `run_lgb_emission_hmm_inference` を追加。`submission.csv` は `sample_submission.csv` order に map し、missing id は `strict_sample_ids=true` で失敗させる。
- Kaggle inference sources: exp072 train, exp148 train, exp099 train, exp111 train, exp112 train。inference metadata は GPU true / internet false。
- local validation so far: exp221 inference source / HMM smoother / copied exp148 helpers の `py_compile` pass、`config.yaml` YAML load pass。

```bash
python3 scripts/prepare_kaggle_notebooks.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148 --notebook inference --kernel-id kentookumura/exp221-lgb-oof-gaussian-emission-hmm-exp148-inference --title "exp221 lgb oof gaussian emission hmm exp148 inference" --run-on-push --strict
make push-kaggle-infer EXP=exp221_lgb_oof_gaussian_emission_hmm_on_exp148
python3 scripts/prepare_kaggle_notebooks.py --experiment exp221_lgb_oof_gaussian_emission_hmm_on_exp148 --notebook inference --kernel-id kentookumura/exp221-lgb-hmm-exp148-infer --title "exp221 lgb hmm exp148 infer" --run-on-push --strict
make push-kaggle-infer EXP=exp221_lgb_oof_gaussian_emission_hmm_on_exp148
kaggle kernels status kentookumura/exp221-lgb-hmm-exp148-infer
kaggle kernels logs kentookumura/exp221-lgb-hmm-exp148-infer
kaggle kernels output kentookumura/exp221-lgb-hmm-exp148-infer -p /tmp/kaggle-output/exp221_inference_v1
.venv/bin/python scripts/validate_submission.py --submission /tmp/kaggle-output/exp221_inference_v1/submission.csv
```

- first inference push result: Kaggle API 400 Bad Request。原因は inference slug 53 chars が Kaggle slug limit 付近だった可能性が高い。train slug は 49 chars で通っていたため、short slug `kentookumura/exp221-lgb-hmm-exp148-infer` に変更。
- inference package validation: package-side `py_compile` pass、ruff F821 pass、metadata `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`, kernel sources exp072/148/099/111/112。
- v1 push result: Kernel version 1 pushed。URL `https://www.kaggle.com/code/kentookumura/exp221-lgb-hmm-exp148-infer`。
- v1 status: COMPLETE。
- exp148 proxy current-test generation: 14,151 rows / 3 wells、15 boosters、294 features、fallback 0、prediction SHA `9a5f5d1030c357d8059c3c9ee2ba3a0578563ce11b9d02fe07906aa8b235d50b`。
- HMM inference: candidate `hmm_lgb_exp148_lgb_mean_s2000_l0500`、14,151 rows / 3 wells、submission rows 14,151、predicted rows 14,151、fallback rows 0、extra prediction ids 0。
- HMM prediction range: min 11598.3203125、max 12235.1181640625、mean 11904.820938974057、std 277.524176688654。
- HMM posterior std: mean 0.6969178318977356、p90 1.2437114715576172。
- SHA: submission `d90926bc87268285640863ddc3e24fbaa4d715c1b7394f7410a2d4f6d13b7cc3`、predictions decompressed `0dde4df77027dcab986201b18aecf9edb8be2e78b43bb7bfe301621d86b684c4`、summary `46955430075abc43c0fb40aca1e6b9639a20c16a685dc9a54695de1af1b075a8`。
- submit-check: PASS。`submission.csv` は `(14151, 2)`、columns `id,tvt`、欠損 0、duplicated id 0。
- note: 見えている test は sample なので、row/well count や prediction range は score checklist ではなく形式確認。score 判断は code submission 後の LB で行う。

## 2026-07-09 submission v1

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
```

- user reported scoring complete: Public LB 7.953。
- Kaggle CLI confirmation: ref `54490473`, date `2026-07-09 10:25:25.300000`, status `SubmissionStatus.COMPLETE`, publicScore `7.953`。
- submitted file SHA: `d90926bc87268285640863ddc3e24fbaa4d715c1b7394f7410a2d4f6d13b7cc3` from `/tmp/kaggle-output/exp221_inference_v1/submission.csv`。
- comparison: exp148 GPU inference v7 7.960 より -0.007 改善。ただし exp193 7.946 より +0.007、exp148 CPU runtime 7.921 より +0.032、exp218 ML anchor 7.843 より +0.110、exp082 ensemble anchor 7.601 より +0.352 悪化。
- decision: train-side OOF の大きな改善は LB に十分転移しなかったため、現時点では採用しない。fixed sigma の点予測 emission を直接提出するより、次にやるなら quantile band / uncertainty-calibrated sigma を別実験で検証する。
