# exp218_gr_wavelet_rotation_confidence_features_on_exp148 セッションノート

## 2026-07-07 実装

### 狙い

`KAGGLE_DIRECTION.md` の `gr_wavelet_rotation_confidence_features_on_exp148`
backlog を実装する。exp167/189/216 では GR denoise / calibration を direct matching
や PF/Beam generation に使う方針は弱かったため、本実験では DWT/FFT/denoise signal を
exp148 learned-likelihood ML surface の add-only confidence feature として評価する。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照: `exp167_fft_denoised_gr_matching_audit`, `exp189_denoised_gr_pfbeam_generation_audit`, `exp216_affine_shift_landscape_ruler_readout`, `exp214_public_raw_gr_residual_scale_control`
- route: `ml_model`
- active variant: `gr_wavelet_rotation_confidence_addonly`
- control retraining: なし。exp148 の保存済み CV / Public LB を historical baseline として参照する。
- inference / submit: 初期実装では対象外。

### Feature

`gr_wavelet_rotation_confidence_features_on_exp148.py` で、raw train の horizontal/typewell GR と known `TVT_input` prefix から target-free feature を再生成する。

- DWT: `pywt` が利用可能なら db4 level 3 approximation/detail residual。未利用環境では rolling mean fallback を記録する。
- FFT: well 全体の detrended spectrum から dominant frequency、dominant energy ratio、rotation-band energy ratio、high-frequency ratio、notch residual ratio を作る。
- local quality: raw/std、DWT detail energy、raw-vs-rolling / raw-vs-SG / raw-vs-DWT absolute residual と rolling correlation。
- candidate observation consistency: 既存 candidate TVT (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) を typewell GR 上で local observation cost として評価し、entropy、default candidate rank、zero-candidate rank proxy、raw-vs-denoised NCC/cost gap を特徴化する。
- interaction: DWT energy x candidate spread、FFT rotation ratio x md_since / candidate range、learned likelihood entropy x DWT energy。

leakage 防止として、hidden-tail true TVT、oracle best、true-error rank、OOF absolute error は feature source に使わない。DWT/FFT/denoised GR、candidate TVT、zero-candidate cost は direct replacement、blend、postprocess、hard gate に使わない。

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

- GRWR feature generation に乱数は使わない。
- upstream exp072 / exp145 cache は固定 artifact として読む。
- LightGBM GPU は `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_train.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py
.venv/bin/ruff check experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_train.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py --select F821,F401
uv run python scripts/validate_experiment.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148
```

### 検証結果

- GRWR feature builder smoke: pass。合成 1 well / 30 rows で 76 feature を生成し、全 numeric feature が finite であることを確認。
- `py_compile`: pass
- `ruff --select F821,F401`: pass
- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `uv run python scripts/validate_experiment.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148`: pass

### Kaggle package prepare

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148 --notebook train --kernel-id kentookumura/exp218-gr-wavelet-rotation-exp148-train --title 'exp218 gr wavelet rotation exp148 train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148 --notebook inference --kernel-id kentookumura/exp218-gr-wavelet-rotation-exp148-inference --title 'exp218 gr wavelet rotation exp148 inference' --strict
.venv/bin/python -m py_compile experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/train/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/train/exp218_gr_wavelet_rotation_confidence_features_on_exp148_train.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/train/settings.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference/settings.py
```

- train package: `experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/train`
- inference package: `experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference`
- train kernel id: `kentookumura/exp218-gr-wavelet-rotation-exp148-train`
- inference kernel id: `kentookumura/exp218-gr-wavelet-rotation-exp148-inference`
- package py_compile: pass
- Kaggle push: train version 1 pushed

### Kaggle train push

```bash
make push-kaggle-train EXP=exp218_gr_wavelet_rotation_confidence_features_on_exp148
kaggle kernels pull kentookumura/exp218-gr-wavelet-rotation-exp148-train -p /tmp/kaggle-pull/exp218-gr-wavelet-rotation-exp148-train -m
kaggle kernels status kentookumura/exp218-gr-wavelet-rotation-exp148-train
kaggle kernels logs kentookumura/exp218-gr-wavelet-rotation-exp148-train
```

- pushed version: 1
- pushed at: 2026-07-07
- Kaggle URL: https://www.kaggle.com/code/kentookumura/exp218-gr-wavelet-rotation-exp148-train
- existence check: `kaggle kernels pull ... -m` succeeded.
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: empty. This is expected for running Kaggle notebooks in this environment; do not treat empty logs alone as failure.

## 2026-07-08 Kaggle train v1 完了確認

### コマンド

```bash
kaggle kernels status kentookumura/exp218-gr-wavelet-rotation-exp148-train
kaggle kernels logs kentookumura/exp218-gr-wavelet-rotation-exp148-train
kaggle kernels output kentookumura/exp218-gr-wavelet-rotation-exp148-train -p /tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/train_v1
```

### 実行結果

- status: `KernelWorkerStatus.COMPLETE`
- output: `/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/train_v1`
- elapsed: 14335.658 sec
- rows / wells / features: 3,783,989 / 773 / 380
- feature groups: `projection_correction,u_disagreement,learned_likelihood_confidence,gr_wavelet_rotation_confidence`
- generated GRWR features: 86
- matched candidate TVT specs: 5 / 8
- feature join coverage: pass、drop rows 0、drop wells 0
- model count: 15

### Pooled CV

| model | RMSE TVT | RMSE target | delta vs exp148 same model |
| --- | ---: | ---: | ---: |
| lgb0 | 8.557165712 | 8.557165689 | -0.042620148 |
| lgb1 | 8.512227651 | 8.512227577 | -0.051743470 |
| lgb2 | 8.524447601 | 8.524447736 | +0.014627882 |
| lgb_mean | 8.475793752 | 8.475793767 | -0.025487430 |

`lgb_mean` は exp148 GPU CV 8.501281182 から -0.025487430 改善。exp160 `lgb_mean` 8.463718774 よりは +0.012074978 弱い。

### Artifact SHA

- `lgb_mean` prediction SHA: `6ad4f96c8ed3cb10c301fd17bf02bf2da6363ddc6e0f96649c9738a09c3b11cf`
- summary SHA: `d904a876d5250ec0eb61fce2ee16ff6d07e612a05dd33c582b56d2c17793b594`
- feature schema SHA: `aaf5f13f1e7c5236cd332dcebfdbf98e9c08247465833232e79ce3ff56362b49`
- model manifest SHA: `904570def0d6ad0140f3df95c8bb38f31823295fd191206290e3833b5b2cc237`

### exp148 OOF 差分

exp148 viewer `experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/artifacts/viewer/exp148_lgb_mean_oof_viewer.csv` と exp218 prediction を id で streaming join して比較した。全 3,783,989 rows / 773 wells で id order は一致。

| bucket | rows | exp218 RMSE | exp148 RMSE | delta |
| --- | ---: | ---: | ---: | ---: |
| 000_050 | 38,650 | 0.957634 | 0.978726 | -0.021092 |
| 050_100 | 38,650 | 1.310175 | 1.316981 | -0.006806 |
| 100_250 | 115,950 | 2.094431 | 2.084639 | +0.009791 |
| 250_500 | 193,157 | 3.315459 | 3.298294 | +0.017165 |
| 500_1000 | 385,911 | 4.800747 | 4.792035 | +0.008711 |
| 1000_plus | 3,011,671 | 9.295198 | 9.325405 | -0.030207 |

by-well は 413 wells 改善、360 wells 悪化、median delta -0.026294、mean delta -0.016525。最大改善は `ac6f01d5` -1.890656、最大悪化は `f88ddb26` +4.075520。worst well では `86454a6f` -1.020068、`1b1eba53` -0.553343、`81bf5923` -1.412768 は改善したが、`fb03ae90` +0.982819、`efe96181` +1.434352、`708caea9` +1.090246 は悪化。

### Feature importance

GRWR feature は 86 列中 85 列で mean importance > 0。最上位の `grwr_fft_rotation_ratio_x_log1p_md_since` は mean importance 3654.4 で全体 4 位。その他の上位 GRWR は `grwr_gr_missing_rate`、`grwr_fft_dominant_frequency_norm`、`grwr_fft_high_frequency_ratio`、`grwr_fft_dominant_energy_ratio`、`grwr_fft_rotation_energy_ratio`。

### 判断

train-side は exp148 GPU CV に対して positive。1000+ long-tail と一部 worst wells で改善し、GRWR block の重要度も十分に出た。一方、100-1000 bucket の小幅悪化、一部 worst-well regression、exp160 より弱い CV、exp160 の train/LB mismatch を踏まえ、inference / submit は自動では行わない。exp218 は supported train-side evidence とし、昇格する場合は同じ exp218 内で current-test GRWR feature generation と raw-test parity を追加してから submit 判断する。

## 2026-07-08 Inference 実装と Kaggle 実行

### 実装

ユーザー依頼により、同じ exp218 内で current-test GRWR feature generation と saved-booster inference を追加した。旧 exp160/exp190 系の `sp45` placeholder path は使わず、train と同じ `build_gr_wavelet_rotation_confidence_features()` を raw test `test_dir` に対して呼ぶ。

- selected variant: `gr_wavelet_rotation_confidence_addonly`
- selected mode: `gpu_repro_guard_dp_threads8`
- selected model: `lgb_mean`
- model count: 15 boosters
- direct denoised-GR replacement / blend / postprocess / hard gate: なし

### 検証

```bash
.venv/bin/python -m py_compile experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py
.venv/bin/ruff check experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/settings.py --select F821,F401
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py
uv run python scripts/validate_experiment.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148 --notebook inference --kernel-id kentookumura/exp218-gr-wavelet-rotation-exp148-inference --title 'exp218 gr wavelet rotation exp148 inference' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference/gr_wavelet_rotation_confidence_features_on_exp148.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference/exp218_gr_wavelet_rotation_confidence_features_on_exp148_inference.py experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/kaggle/inference/settings.py
```

- `py_compile`: pass
- `ruff --select F821,F401`: pass
- Jupytext convert / `--test`: pass
- `validate_experiment.py`: pass
- inference package `py_compile`: pass
- inference metadata: `run_on_push=true`, GPU enabled, internet disabled, train source `kentookumura/exp218-gr-wavelet-rotation-exp148-train`

### Kaggle push / 完了確認

```bash
make push-kaggle-infer EXP=exp218_gr_wavelet_rotation_confidence_features_on_exp148
kaggle kernels pull kentookumura/exp218-gr-wavelet-rotation-exp148-inference -p /tmp/kaggle-pull/exp218-gr-wavelet-rotation-exp148-inference -m
kaggle kernels status kentookumura/exp218-gr-wavelet-rotation-exp148-inference
kaggle kernels logs kentookumura/exp218-gr-wavelet-rotation-exp148-inference
kaggle kernels output kentookumura/exp218-gr-wavelet-rotation-exp148-inference -p /tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1
```

- pushed version: 1
- pushed at: 2026-07-08
- URL: https://www.kaggle.com/code/kentookumura/exp218-gr-wavelet-rotation-exp148-inference
- final status: `KernelWorkerStatus.COMPLETE`
- output: `/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1`
- elapsed: 108.982 sec

### Inference metrics

- selected: `gr_wavelet_rotation_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`
- model count: 15
- feature count: 380
- test rows: 14,151
- submission rows: 14,151
- predicted rows: 14,151
- fallback rows: 0
- prediction min / max: 11590.19921875 / 12239.5390625
- prediction mean / std: 11905.245071916848 / 278.60856874126875
- prediction SHA: `483845c8969e99e8d12c9dfcbe43bb8dfc727a1df8905ef045f02e35ebdcbff1`
- submission SHA: `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`
- summary SHA: `4c6cecf940b5793efe3b496d2e668f8d2116fea5299d7663e90d4939da3368c0`
- inference feature schema SHA: `e70f65f05b865cc77f426b1671a01714e597a1795c857eef4b7c17323e1344d6`
- raw-test learned likelihood decompressed SHA: `8d1146ac1e68da67a2c8d2d00788c1593fc99654b949e0a5ac065cf781344e13`
- raw-test long likelihood decompressed SHA: `92ae5e9328073ac2727fa18dc2e03025a557e9646b8b9be57fd051d1ae86c612`
- anchor T0 vs last_known_tvt max abs diff: 0.0
- known prefix rows min / max: 1442 / 2083
- GRWR current-test generated features: 86

### Submit check

```bash
python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1/submission.csv --sample data/raw/sample_submission.csv
```

- status: PASS
- rows: 14,151 / sample 14,151
- columns: `id`, `tvt`
- header matches sample: yes
- ID order matches sample: yes
- duplicate IDs: 0
- missing / NaN / Inf-like values: 0

### 判断

Inference artifact は ready。current-test GRWR feature replay、exp145 learned likelihood current-test generation、15 saved boosters の `lgb_mean` prediction、submission format check はすべて成立した。ただし train-side の懸念は残るため、このターンでは submit しない。

## 2026-07-08 Submission scoring 完了

ユーザーから提出と scoring 完了の連絡があり、Kaggle submissions table を確認した。inference v1 直後の blank-description submission を exp218 として記録する。

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp218_gr_wavelet_rotation_confidence_features_on_exp148 --file /tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/inference_v1/submission.csv --version v059 --cv 8.475793752 --public-lb 7.843 --private-lb - --notes "ref=54457577; kernel=kentookumura/exp218-gr-wavelet-rotation-exp148-inference v1; selected=gr_wavelet_rotation_confidence_addonly/gpu_repro_guard_dp_threads8/lgb_mean; current-test GRWR replay; 15 boosters; fallback_rows=0; submit-check PASS; improves exp148 CPU runtime Public LB 7.921 by -0.078 and exp148 GPU 7.960 by -0.117; updates ML route submitted anchor, but remains worse than ensemble anchor exp082 7.601"
```

- ref: `54457577`
- submitted at: 2026-07-08 09:47:59.040000 UTC / 2026-07-08 18:47:59.040000 JST
- status: `SubmissionStatus.COMPLETE`
- Public LB: 7.843
- submission SHA: `77a2c2804749dc811ba61f43d9d8827c69282e83e116233559da80b6820c0824`
- delta vs exp148 CPU runtime Public LB 7.921: -0.078
- delta vs exp148 GPU inference v7 Public LB 7.960: -0.117
- delta vs exp198 Public LB 7.930: -0.087
- delta vs exp160 Public LB 8.061: -0.218
- delta vs exp082 ensemble Public LB 7.601: +0.242

### 判断更新

exp218 は exp148 CPU runtime anchor 7.921 を上回ったため、ML route submitted anchor として採用する。overall では exp082 ensemble Public LB 7.601 が引き続き最良。
