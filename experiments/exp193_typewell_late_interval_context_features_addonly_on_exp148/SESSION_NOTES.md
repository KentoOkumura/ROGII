# exp193_typewell_late_interval_context_features_addonly_on_exp148 セッションノート

## 2026-07-04 実装

### 狙い

`KAGGLE_DIRECTION.md` の `typewell_late_interval_context_features_addonly_on_exp148` backlog を `exp193_typewell_late_interval_context_features_addonly_on_exp148` として実装する。

exp174 で ML 予測の late-range hard clip / shrink は悪化し、exp176 では candidate ranker に `candidate_pct` / `known_last_pct` 系 signal を入れると positive だった。今回は candidate 別 feature を使わず、raw typewell range と observed prefix last-known TVT だけから作る well-context feature が exp148 ML anchor に効くかを小さく見る。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 参照: `exp174_typewell_late_range_ml_posthoc_clip_audit`, `exp176_typewell_late_range_pfbeam_candidate_prior`, `exp191_typewell_late_range_continuity_selector_on_exp176`
- route: `ml_model`
- active variant: `typewell_late_interval_context_addonly`
- control retraining: なし。exp148 の保存済み CV / Public LB を historical baseline として参照する。
- inference / submit: 初期実装では対象外。

### Feature

`typewell_late_interval_context_features_addonly_on_exp148.py` で、raw train の `*__typewell.csv` と observed `TVT_input` prefix anchor から target-free feature を生成する。

- feature prefix: `tlic_`
- thresholds: late50 / late60 / late70
- 追加列: typewell min/max/span、late interval min/max/span、`known_last_pct`、`known_last_to_lateXX_min_delta`、`known_last_inside_lateXX`
- 除外: `candidate_pct_*`、candidate 別 violation、PF/Beam/ML prediction direct replacement、clip、blend、postprocess、hard selector

leakage 防止として、hidden-tail true TVT、oracle best、true-error rank、OOF absolute error は feature source に使わない。

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

- typewell context feature generation に乱数は使わない。
- raw typewell file set hash を train summary / manifest に記録する。
- upstream exp072 / exp145 cache は固定 artifact として読む。
- LightGBM GPU は `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, `n_jobs/num_threads=8`。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/settings.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/settings.py --select F821,F401
uv run python scripts/validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821,F401`: pass
- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148`: pass
- raw-only feature builder smoke: pass。3 wells x 2 rows の synthetic frame で 19 features、context 773 wells、summary 4 rows。
- exp072 local cache subset smoke: skipped。`experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` がローカルに無く、Kaggle input 前提のため。

### Kaggle package prepare

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --notebook train --kernel-id kentookumura/exp193-typewell-late-interval-context-features-addonly-exp148-train --title 'exp193 typewell late interval context features addonly exp148 train' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/settings.py
```

- train package: `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train`
- train kernel id: `kentookumura/exp193-typewell-late-interval-context-features-addonly-exp148-train`
- package py_compile: pass
- Kaggle push: not run

## 2026-07-04 Kaggle train v1 実行

ユーザー依頼により、生成済み train package を Kaggle GPU で実行する。

実行前 GPU コスト確認:

- active variants: 1 (`typewell_late_interval_context_addonly`)
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- control / parent retraining: なし
- GPU: enabled (`enable_gpu=true`, LightGBM `gpu_use_dp=true`, deterministic flags, threads=8)

実行予定:

```bash
make push-kaggle-train EXP=exp193_typewell_late_interval_context_features_addonly_on_exp148
```

Kaggle kernel:

- `kentookumura/exp193-typewell-late-interval-context-features-addonly-exp148-train`

初回 push 結果:

```bash
make push-kaggle-train EXP=exp193_typewell_late_interval_context_features_addonly_on_exp148
```

- result: failed
- error: `400 Client Error: Bad Request ... SaveKernel`
- reason: Kaggle API から詳細なし。67 文字の長い slug が metadata 制約に当たった可能性があるため、同じ exp193 のまま最近の repo 実験と同じ短い meaningful slug に再 prepare する。
- revised kernel id: `kentookumura/exp193-typewell-late-context-exp148-train`

再 prepare:

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --notebook train --kernel-id kentookumura/exp193-typewell-late-context-exp148-train --title 'exp193 typewell late context exp148 train' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/exp193_typewell_late_interval_context_features_addonly_on_exp148_train.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/train/settings.py
make push-kaggle-train EXP=exp193_typewell_late_interval_context_features_addonly_on_exp148
```

再 push 結果:

- result: success
- Kaggle version: 1
- URL: <https://www.kaggle.com/code/kentookumura/exp193-typewell-late-context-exp148-train>
- status check: `KernelWorkerStatus.RUNNING`
- logs: empty while running. この環境では実行中に CLI logs が空で返る前提のため、完了後に同じ kernel id で再取得する。

記録更新:

```bash
make update-summary
uv run python scripts/validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148
kaggle kernels status kentookumura/exp193-typewell-late-context-exp148-train
```

- `metrics.json`: `running_kaggle_train_v1`
- `experiment_summary.md`: `running_kaggle_train_v1`
- `validate_experiment.py`: pass
- latest status: `KernelWorkerStatus.RUNNING`

## 2026-07-05 Kaggle train v1 完了確認

ユーザー通知を受けて Kaggle status / logs を再取得した。

```bash
kaggle kernels status kentookumura/exp193-typewell-late-context-exp148-train
kaggle kernels logs kentookumura/exp193-typewell-late-context-exp148-train > /tmp/exp193_kaggle_logs.json
mkdir -p experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/train_v1 && kaggle kernels output kentookumura/exp193-typewell-late-context-exp148-train -p experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output dir: `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/train_v1`
- output size: 446M
- reason for output download: logs で CV は確認できたが、feature importance と model manifest / model SHA を確認するため。

CV:

| model | pooled RMSE TVT | exp148 同 config 差分 | prediction SHA |
| --- | ---: | ---: | --- |
| `lgb0` | 8.553543816914422 | -0.046242042464468724 | `3d34f51cb77d70a7fb1ed97bef3b48ff509c9f57bdf1f24c3af4a5a78c72b59a` |
| `lgb1` | 8.475340902293846 | -0.08863021893582257 | `3430c2e47546843c44cc271f4d4d625842ef9fcf35f0014e039a2f2e1919ee54` |
| `lgb2` | 8.510015020743442 | +0.00019530194936656642 | `7bf5746d0c9e828952d5f9e315de09240c6e303827176ad7d8d810fb15ac5618` |
| `lgb_mean` | 8.456665438542778 | -0.04461574335304164 | `3a314436d8e4e291eb6d63607d06d27a6ecb05735827111cd24aa49b41d59d04` |

Fold metrics:

- `lgb0`: fold0 8.825960313 / iter4998, fold1 8.828045975 / iter404, fold2 7.634608195 / iter737, fold3 8.285296927 / iter258, fold4 9.112826558 / iter324
- `lgb1`: fold0 8.622115174 / iter9999, fold1 8.507907469 / iter5335, fold2 7.556387638 / iter2289, fold3 8.508163402 / iter656, fold4 9.107037512 / iter1411
- `lgb2`: fold0 8.666423684 / iter9994, fold1 8.525558639 / iter5111, fold2 7.575663217 / iter2403, fold3 8.593319026 / iter681, fold4 9.114113451 / iter1282

Feature / artifact checks:

- rows / wells / features: 3,783,989 / 773 / 313
- feature join coverage: pass、dropped base rows 0、dropped base wells 0
- typewell context features: 19、context rows 773、missing rate max 0.0
- typewell file set SHA256: `a9ada5643bbd2775adab48cc53872610868eae8fae313eb5a23ba60b049e8938`
- model count: 15
- model manifest SHA256: `e8336b7a2058e584219750b26b30cb582802cf9b030e9ed728346230cd7d1e67`
- feature schema SHA256: `c762e6987be934ce6145ba954e2f60e68c5c36c7a54b2912253d610aac16fd80`
- summary JSON SHA256: `80d1c1871c1a6c9830952a5799c743bfb76dcd92117719918e781cbc44162c6b`
- prediction gzip SHA256: `0a613e0b9ee06a65247a02a401b5be58e8d57b10931aabdc1b9bf6df5edd4676`
- prediction decompressed SHA256: `c171a0655ff3011e198d8b5ad1c74c5d3a8b9f086b09cd47dd613385133721dc`

Feature importance:

- `tlic_known_last_pct`: rank 46 / 313、mean importance 1987.133333
- `tlic_known_last_to_late70_min_delta`: rank 89、mean importance 1235.0
- `tlic_known_last_to_late50_min_delta`: rank 109、mean importance 1068.466667
- `tlic_known_last_to_late60_min_delta`: rank 114、mean importance 1044.733333
- `tlic_typewell_min`: rank 121、mean importance 992.2

Bucket / worst-well readout:

- distance `000_050`: RMSE 1.001955
- distance `1000_plus`: RMSE 9.273843
- worst wells remain `86454a6f` 47.865982, `1b1eba53` 46.223045, `fb03ae90` 44.735210.

Interpretation:

- train-side supported。exp148 `lgb_mean` 8.50128118189582 から -0.04461574335304164 改善。
- exp160 は CV positive でも Public LB negative だったため、この CV だけで submit しない。
- 次は同じ exp193 内で current-test typewell context feature generation を inference notebook に実装し、raw-test/current-test parity、fallback 0、submit-check を確認する。

## 2026-07-05 inference port 作成

ユーザー依頼により、同じ exp193 内で saved-booster inference を作成した。

実装内容:

- `typewell_late_interval_context_features_addonly_on_exp148.py` に `run_saved_model_inference()` を追加。
- exp148 の saved-booster inference flow をベースに、raw current-test replay、exp092 U-projection、exp145 learned-likelihood feature generation、exp193 `tlic_` 19 features を同じ feature contract で再生成する。
- exp193 train v1 manifest の `typewell_context_feature_groups` と current-test generated group を完全一致チェックする。
- selected contract は `typewell_late_interval_context_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean`。train v1 の 15 saved boosters を平均する。
- direct typewell clip、candidate hard selector、blend、postprocess は追加しない。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py --select F821,F401
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
uv run python scripts/validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --notebook inference --kernel-id kentookumura/exp193-typewell-late-context-exp148-inference --title 'exp193 typewell late context exp148 inference' --strict
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/settings.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.ipynb --select F821,F401
```

結果:

- module / notebook py_compile: pass
- `ruff --select F821,F401`: pass
- Jupytext inference convert / `--test`: pass
- `validate_experiment.py`: pass
- inference package: `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference`
- inference kernel id: `kentookumura/exp193-typewell-late-context-exp148-inference`
- metadata: GPU enabled、internet off、`run_on_push=false`
- kernel sources: exp072 full replay cache train、exp193 train v1、exp099、exp111、exp112

未実施:

- Kaggle inference push / run
- inference output の feature schema parity、generated feature count 19、fallback rows 0、sample submission 互換、prediction/submission SHA
- submit-check / 提出

## 2026-07-05 Kaggle inference v1/v2 実行

ユーザー依頼により、exp193 inference を Kaggle で実行した。

### v1

Package を `run_on_push=true` にして push:

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --notebook inference --kernel-id kentookumura/exp193-typewell-late-context-exp148-inference --title 'exp193 typewell late context exp148 inference' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/settings.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.ipynb --select F821,F401
make push-kaggle-infer EXP=exp193_typewell_late_interval_context_features_addonly_on_exp148
kaggle kernels status kentookumura/exp193-typewell-late-context-exp148-inference
kaggle kernels logs kentookumura/exp193-typewell-late-context-exp148-inference
```

- kernel: `kentookumura/exp193-typewell-late-context-exp148-inference`
- version: 1
- status: `KernelWorkerStatus.ERROR`
- failure: current-test learned-likelihood feature generation fallback に入った後、`generator.candidates must not be empty`。
- cause: exp193 config に exp145/exp148 と同じ `generator.candidates` block が無かった。
- fix: exp145/exp148 と同じ generator block を exp193 `config.yaml` に追加。これは learned-likelihood current-test regeneration の設定で、exp193 の feature hypothesis は変えていない。

### v2

修正後に validation / package prepare / push:

```bash
uv run python scripts/validate_experiment.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.py --select F821,F401
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --notebook inference --kernel-id kentookumura/exp193-typewell-late-context-exp148-inference --title 'exp193 typewell late context exp148 inference' --run-on-push --strict
.venv/bin/python -m py_compile experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/settings.py
.venv/bin/ruff check experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/typewell_late_interval_context_features_addonly_on_exp148.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference/exp193_typewell_late_interval_context_features_addonly_on_exp148_inference.ipynb --select F821,F401
make push-kaggle-infer EXP=exp193_typewell_late_interval_context_features_addonly_on_exp148
```

結果:

- version: 2
- status: `KernelWorkerStatus.COMPLETE`
- URL: <https://www.kaggle.com/code/kentookumura/exp193-typewell-late-context-exp148-inference>
- elapsed_seconds: 116.88
- train manifest: `/kaggle/input/notebooks/kentookumura/exp193-typewell-late-context-exp148-train/artifacts/exp193_typewell_late_interval_context_features_addonly_on_exp148_lgb_models/manifest.json`
- selected saved boosters: 15
- raw-test learned likelihood features: generated current-test features, 14,151 rows / 51 columns
- raw-test typewell context: 3 context wells, generated 19 features
- output dir: `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/inference_v2`

Metrics:

| item | value |
| --- | ---: |
| feature_count | 313 |
| base_feature_count | 196 |
| projection_feature_count | 69 |
| learned_likelihood_feature_count | 54 |
| typewell_context_feature_count | 19 |
| test_rows | 14,151 |
| submission_rows | 14,151 |
| predicted_rows | 14,151 |
| fallback_rows | 0 |
| prediction_min | 11590.3720703125 |
| prediction_max | 12240.1171875 |
| prediction_mean | 11905.43199423296 |
| prediction_std | 278.79483926833177 |

SHA:

- prediction SHA256: `3567ebd4e48b1ab08e3b2ebf05dfa5061c65303e6f9081be262edd7940cbd0f8`
- submission SHA256: `9265e3e19e7eea20c6e0097b3b581b4a15c29353ebb77875d09ac30475502695`
- current-test learned likelihood features decompressed SHA256: `8d1146ac1e68da67a2c8d2d00788c1593fc99654b949e0a5ac065cf781344e13`
- current-test learned likelihood long decompressed SHA256: `92ae5e9328073ac2727fa18dc2e03025a557e9646b8b9be57fd051d1ae86c612`

Checks:

```bash
kaggle kernels output kentookumura/exp193-typewell-late-context-exp148-inference -p experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/inference_v2
.venv/bin/python .agents/skills/kaggle-submit-check/scripts/check_submission.py experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/inference_v2/submission.csv --sample data/raw/sample_submission.csv
```

- submit-check: PASS、fail 0、warn 0
- sample submission header / row count: match
- sample id order: match
- duplicate ID / NaN / Inf: none
- train manifest feature schema vs inference feature schema: exact match
- train typewell group count 19、inference typewell schema count 19

提出 / スコアリング:

ユーザー通知後に Kaggle submissions を確認し、scoring 完了を記録した。

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
.venv/bin/python scripts/record_submission.py --experiment exp193_typewell_late_interval_context_features_addonly_on_exp148 --file experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/output/inference_v2/submission.csv --version v051 --cv 8.456665439 --public-lb 7.946 --private-lb - --notes "ref=54347471; kernel=kentookumura/exp193-typewell-late-context-exp148-inference v2; selected=typewell_late_interval_context_addonly/gpu_repro_guard_dp_threads8/lgb_mean; 15 boosters; tlic_19; fallback_rows=0; feature schema exact match; submit-check PASS; improves exp148 Public LB 7.960 by -0.014"
```

- ref: `54347471`
- submitted at: 2026-07-05 02:12:58.030000 UTC / 2026-07-05 11:12:58.030000 JST
- status: `SubmissionStatus.COMPLETE`
- Public LB: 7.946
- Private LB: 未表示
- submission SHA256: `9265e3e19e7eea20c6e0097b3b581b4a15c29353ebb77875d09ac30475502695`

exp148 GPU inference v7 Public LB 7.960 からは -0.014 改善した。一方、ユーザー確認済みの exp148 CPU runtime submission Public LB 7.921 (`ref=54183122`) には +0.025 届かないため、exp193 は ML route submitted anchor には採用しない。CV 改善量 -0.044615743 より LB 改善量は小さく、CV-to-LB 転移は控えめだった。アンサンブル route anchor の exp082 Public LB 7.601 は引き続き全体最良。
