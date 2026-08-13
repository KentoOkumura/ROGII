# exp190_denoised_calibrated_matching_features_on_exp148 セッションノート

## 2026-07-04 実装

### 狙い

`backlog/KAGGLE_DIRECTION.md` の `denoised_calibrated_matching_features_on_exp148` backlog を `exp190_denoised_calibrated_matching_features_on_exp148` として実装する。

exp167 では FFT notch は弱く、rolling median / Savitzky-Golay smoothing は matching surface の gap / entropy / decoy gap を改善した。exp170 の heel calibration は不採用、exp171 の posterior candidate direct replacement も不採用だった。したがって本実験では direct TVT 置換をせず、raw/smoothed GR shift-scan の surface sharpness / posterior ambiguity / candidate disagreement を exp148 の ML route anchor に add-only feature として渡す。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照 audit: `exp167_fft_denoised_gr_matching_audit`, `exp170_heel_calibrated_shift_scan_pfbeam_audit`, `exp171_bimodal_posterior_pfbeam_candidate_audit`
- route: `ml_model`
- active variant: `denoised_calibrated_matching_addonly`
- control retraining: なし。exp148 の保存済み CV / Public LB を historical baseline として参照する。
- inference / submit: 初期実装では対象外。

### Feature

`denoised_calibrated_matching_features_on_exp148.py` で、raw train の horizontal/typewell GR と known `TVT_input` prefix から target-free feature を再生成する。

- filters: `raw`, `rolling_median_11`, `savgol_31_p2`
- shift grid: -80ft から +80ft、4ft step
- local offsets: `[-36, -24, -12, -6, 0, 6, 12, 24, 36]`
- full-row coverage: well 内 16 row 間隔の deterministic scan grid を走査し、同一 well 内で全行へ補間する
- 追加列: top1/top2 cost, top1/top2 gap, entropy, top1/top2 shift, top2 TVT gap, posterior p/entropy, posterior-minus-likpf, raw-vs-smoothed localization movement, candidate spread/range, distance interactions, known-prefix backtest quality

leakage 防止として、hidden-tail true TVT、oracle best、true-error rank、abs error は feature source に使わない。prefix backtest は観測済み `TVT_input` prefix 内だけで計算する。FFT notch と heel calibration 列は入れない。

### Kaggle train 実行前確認

- active variants: 1
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- train notebook split: なし
- control / parent retraining: なし
- GPU: enabled

### 再現性メモ

- GR shift-scan feature generation に乱数は使わない。
- upstream exp072 / exp145 cache は固定 artifact として読む。
- LightGBM GPU は `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp190_denoised_calibrated_matching_features_on_exp148/denoised_calibrated_matching_features_on_exp148.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_train.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_inference.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/settings.py
.venv/bin/ruff check experiments/exp190_denoised_calibrated_matching_features_on_exp148/denoised_calibrated_matching_features_on_exp148.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_train.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/exp190_denoised_calibrated_matching_features_on_exp148_inference.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/settings.py --select F821,F401
uv run python scripts/validate_experiment.py --experiment exp190_denoised_calibrated_matching_features_on_exp148
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821,F401`: pass
- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `uv run python scripts/validate_experiment.py --experiment exp190_denoised_calibrated_matching_features_on_exp148`: pass
- `uv run python scripts/update_experiment_summary.py`: pass。`experiment_summary.md` に exp190 を反映。

### Kaggle package prepare

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp190_denoised_calibrated_matching_features_on_exp148 --notebook train --kernel-id kentookumura/exp190-denoised-calibrated-matching-exp148-train --title 'exp190 denoised calibrated matching exp148 train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp190_denoised_calibrated_matching_features_on_exp148 --notebook inference --kernel-id kentookumura/exp190-denoised-calibrated-matching-exp148-inference --title 'exp190 denoised calibrated matching exp148 inference' --strict
.venv/bin/python -m py_compile experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/train/denoised_calibrated_matching_features_on_exp148.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/train/exp190_denoised_calibrated_matching_features_on_exp148_train.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/train/settings.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/inference/exp190_denoised_calibrated_matching_features_on_exp148_inference.py experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/inference/settings.py
```

- train package: `experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/train`
- inference package: `experiments/exp190_denoised_calibrated_matching_features_on_exp148/kaggle/inference`
- train kernel id: `kentookumura/exp190-denoised-calibrated-matching-exp148-train`
- inference kernel id: `kentookumura/exp190-denoised-calibrated-matching-exp148-inference`
- package py_compile: pass
- Kaggle push: train version 1 pushed
- Kaggle URL: https://www.kaggle.com/code/kentookumura/exp190-denoised-calibrated-matching-exp148-train
- Initial Kaggle status: `KernelWorkerStatus.RUNNING`
- Initial Kaggle logs: empty immediately after push

### Kaggle train push

```bash
make push-kaggle-train EXP=exp190_denoised_calibrated_matching_features_on_exp148
kaggle kernels status kentookumura/exp190-denoised-calibrated-matching-exp148-train
kaggle kernels logs kentookumura/exp190-denoised-calibrated-matching-exp148-train
```

- pushed version: 1
- pushed at: 2026-07-04
- status check result: `RUNNING`
- note: train 実行直後のため logs は未出力。CV / fold metrics は完了後に Kaggle logs または notebook output で確認する。
- recheck after 2 minutes: `RUNNING`, logs still empty

### Kaggle train v1 result

```bash
kaggle kernels status kentookumura/exp190-denoised-calibrated-matching-exp148-train
kaggle kernels logs kentookumura/exp190-denoised-calibrated-matching-exp148-train
```

- final status: `KernelWorkerStatus.COMPLETE`
- source: completed Kaggle logs
- rows / wells: 3,783,989 / 773
- features: 431
- feature groups: `projection_correction,u_disagreement,learned_likelihood_confidence,denoised_calibrated_matching`
- feature join coverage: full pass, dropped rows 0, dropped wells 0
- elapsed seconds: 15,570.458

| model | pooled RMSE TVT | pooled RMSE target | exp148 同 config 比 |
| --- | ---: | ---: | ---: |
| `lgb0` | 8.601678275458363 | 8.601678160524132 | +0.001892416079473 |
| `lgb1` | 8.539624480134293 | 8.539624596538978 | -0.024346641095376 |
| `lgb2` | 8.540073561506953 | 8.540073587431210 | +0.030253842712878 |
| `lgb_mean` | 8.503596159484825 | 8.503596252227380 | +0.002314977589005 |

Best は `lgb_mean` 8.503596159484825。exp148 `lgb_mean` 8.50128118189582 から +0.002314977589005 悪化し、exp160 `lgb_mean` 8.463718773783008 からも +0.039877385701817 悪い。

採否: train-side rejected。`lgb1` 単体は改善したが、採用基準の `lgb_mean` が exp148 を超えないため、current-test feature parity 実装、inference、submit は行わない。

実行中に DCM feature generation で pandas DataFrame fragmentation warning が多数出た。実行は完了しており結果は有効だが、再利用する場合は列追加を一括 concat へ寄せる runtime 改善余地がある。
