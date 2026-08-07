# exp195_denoised_calibrated_matching_replacement_only_on_exp148 セッションノート

## 2026-07-04 実装

### 狙い

`KAGGLE_DIRECTION.md` の `denoised_calibrated_matching_replacement_only_on_exp148` backlog を `exp195_denoised_calibrated_matching_replacement_only_on_exp148` として実装する。

exp190 add-only は採用基準の `lgb_mean` では exp148 をわずかに下回ったが、`lgb1` 単体は exp148 同 config を改善した。add-only で exp145 learned likelihood confidence と DCM signal が競合した可能性を、full block replacement として切り分ける。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照 audit: `exp167_fft_denoised_gr_matching_audit`, `exp170_heel_calibrated_shift_scan_pfbeam_audit`, `exp171_bimodal_posterior_pfbeam_candidate_audit`, `exp190_denoised_calibrated_matching_features_on_exp148`
- route: `ml_model`
- active variant: `denoised_calibrated_matching_replacement_only`
- active feature groups: `projection_correction`, `u_disagreement`, `denoised_calibrated_matching`
- removed feature group: `learned_likelihood_confidence` (`ll_*` 54列)
- control retraining: なし。exp148 の保存済み CV / Public LB を historical baseline として参照する。
- inference / submit: 初期実装では対象外。

### Feature

`denoised_calibrated_matching_replacement_only_on_exp148.py` で、raw train の horizontal/typewell GR と known `TVT_input` prefix から target-free DCM feature を再生成する。

- filters: `raw`, `rolling_median_11`, `savgol_31_p2`
- shift grid: -80ft から +80ft、4ft step
- local offsets: `[-36, -24, -12, -6, 0, 6, 12, 24, 36]`
- full-row coverage: well 内 16 row 間隔の deterministic scan grid を走査し、同一 well 内で全行へ補間する
- DCM 列: top1/top2 cost, top1/top2 gap, entropy, top1/top2 shift, top2 TVT gap, posterior p/entropy, posterior-minus-likpf, raw-vs-smoothed localization movement, candidate spread/range, distance interactions, known-prefix backtest quality

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
- upstream exp072 / exp145 cache は固定 artifact として読む。exp145 cache は replacement boundary / coverage inventory として読み、active model feature には入れない。
- LightGBM GPU は `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/settings.py
.venv/bin/ruff check experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/settings.py --select F821,F401
.venv/bin/python scripts/validate_experiment.py --experiment exp195_denoised_calibrated_matching_replacement_only_on_exp148
```

### 現在の状態

- 実装済み。
- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `py_compile`: pass
- `ruff --select F821,F401`: pass
- `validate_experiment.py --experiment exp195_denoised_calibrated_matching_replacement_only_on_exp148`: pass
- train package strict prepare: pass
- package py_compile: pass
- Kaggle train push は未実行。

### Kaggle package prepare

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp195_denoised_calibrated_matching_replacement_only_on_exp148 --notebook train --kernel-id kentookumura/exp195-dcm-replace-exp148-train --title 'exp195 dcm replace exp148 train' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/settings.py
```

- train package: `experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train`
- train kernel id: `kentookumura/exp195-dcm-replace-exp148-train`
- train title: `exp195 dcm replace exp148 train`

## 2026-07-05 再検証

ユーザー依頼 `denoised_calibrated_matching_replacement_only_on_exp148` 実装確認として、既存 exp195 実装を確認した。

- active feature groups は `projection_correction`, `u_disagreement`, `denoised_calibrated_matching`。
- `learned_likelihood_confidence` (`ll_*`) は coverage / replacement boundary inventory として生成されるが、active model feature list には入らない。
- train notebook は helper `main()` の薄い呼び出しではなく、設定確認、入力 preview、replacement-only train、metrics 表示のセル構成。
- inference notebook は初期状態では submission を作らず、train-side CV が positive の場合だけ同じ exp195 内で current-test DCM feature generation と saved-booster inference を追加する方針。
- Kaggle train package metadata は `kentookumura/exp195-dcm-replace-exp148-train`、GPU on、internet off、kernel sources は exp072/exp145 のみ。

再実行した検証:

```bash
.venv/bin/python -m py_compile experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/settings.py
.venv/bin/ruff check experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/settings.py --select F821,F401
.venv/bin/python scripts/validate_experiment.py --experiment exp195_denoised_calibrated_matching_replacement_only_on_exp148
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/exp195_denoised_calibrated_matching_replacement_only_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/denoised_calibrated_matching_replacement_only_on_exp148.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/exp195_denoised_calibrated_matching_replacement_only_on_exp148_train.py experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train/settings.py
```

結果:

- `py_compile`: pass
- `ruff --select F821,F401`: pass
- `validate_experiment.py`: pass
- Jupytext train / inference `--test`: pass
- prepared Kaggle train package `py_compile`: pass
- `__file__` / thin `main()` notebook guard: no matches

現在の状態:

- 実装済み。
- Kaggle train push は未実行。
- 予定コストは 1 active variant x 1 mode x 3 LightGBM configs x 5 folds = 15 boosters。
- parent/control 再学習は含まない。

## 2026-07-05 Kaggle train push

ユーザー依頼により Kaggle train を実行する。

実行前ガード:

- active variants: 1 (`denoised_calibrated_matching_replacement_only`)
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- GPU: enabled
- control / parent retraining: なし

Kaggle package:

- package dir: `experiments/exp195_denoised_calibrated_matching_replacement_only_on_exp148/kaggle/train`
- kernel id: `kentookumura/exp195-dcm-replace-exp148-train`
- title: `exp195 dcm replace exp148 train`
- run_on_push: true
- internet: off
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: `kentookumura/exp072-exp063-full-replay-feature-cache-train`, `kentookumura/exp145-train`

実行コマンド:

```bash
make push-kaggle-train EXP=exp195_denoised_calibrated_matching_replacement_only_on_exp148
```

実行結果:

- `Kernel version 1 successfully pushed.`
- URL: `https://www.kaggle.com/code/kentookumura/exp195-dcm-replace-exp148-train`
- `kaggle kernels pull kentookumura/exp195-dcm-replace-exp148-train -p /tmp/kaggle-pull/exp195-dcm-replace-exp148-train -m`: success
- `kaggle kernels logs kentookumura/exp195-dcm-replace-exp148-train`: warning のみで logs は空。実行中 logs が空のまま返る既知挙動として扱い、失敗判定しない。
- `kaggle kernels status kentookumura/exp195-dcm-replace-exp148-train`: `KernelWorkerStatus.RUNNING`

現在の状態:

- Kaggle train v1 running。
- CV / 生成物 / SHA は未確定。
- 完了後は logs から pooled metrics、fold metrics、feature count、生成物パスを記録する。

### 完了確認

```bash
kaggle kernels status kentookumura/exp195-dcm-replace-exp148-train
kaggle kernels logs kentookumura/exp195-dcm-replace-exp148-train > /tmp/exp195_kaggle_logs.json
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- logs source: `/tmp/exp195_kaggle_logs.json`
- logs に summary JSON と pooled metrics が出力されたため、Kaggle output archive は取得していない。
- 実行中に `PerformanceWarning: DataFrame is highly fragmented` が多数出たが、notebook は完了し、metrics / manifest / feature importance plot まで生成した。

CV:

| model | RMSE TVT | delta vs exp148 | delta vs exp190 add-only | prediction SHA256 |
| --- | ---: | ---: | ---: | --- |
| lgb0 | 9.612543035441323 | +1.0127571760624328 | +1.0108647599829599 | `f455fcf18496820e5da96c27bea9cf680c2b8a86fafd134d075a528ef5c7dd53` |
| lgb1 | 9.405030561019409 | +0.8410594397897402 | +0.8654060808851156 | `a871ccfa10f14f37b7000f6bfc4f045b9e8136d90a38cf53d7c1081c2cc34f3e` |
| lgb2 | 9.388749748145075 | +0.8789300293510003 | +0.8486761866381229 | `65c3c269a223bd6328fd53d2665766e5daba7ada6711128fff1d27e47c6e0f98` |
| lgb_mean | 9.409612610766938 | +0.9083314288711186 | +0.9060164512821132 | `cab0cd29913e8cae39c807f3588e170a93b3590f7ad90f0601dc048af624fb76` |

Feature / coverage:

- rows: 3,783,989
- wells: 773
- features: 377
- active feature groups: `projection_correction,u_disagreement,denoised_calibrated_matching`
- removed feature group: `learned_likelihood_confidence`
- feature join coverage: pass
- dropped rows / wells: 0 / 0
- elapsed seconds: 15,193.865

判断:

- completed_train_side_rejected_no_submit。
- exp148 `lgb_mean` 8.50128118189582 から +0.9083314288711186 悪化。
- exp190 add-only `lgb_mean` 8.503596159484825 から +0.9060164512821132 悪化。
- replacement-only DCM block は exp145 learned likelihood confidence block の代替にならない。
- current-test feature generation、inference port、submit は行わない。
