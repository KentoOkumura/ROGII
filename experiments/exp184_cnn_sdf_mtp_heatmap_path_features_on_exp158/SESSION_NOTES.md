# exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 セッションノート

## 2026-07-03 実装

`cnn_sdf_mtp_heatmap_path_features_on_exp158` backlog を実装する。exp182 では full-fold `base_real_w128_b64_fullfold` が top3 within10 0.500000、shuffled 0.218536、no-GR 0.071429 と real GR signal を確認した。一方 worst-well top3 0.0 が残るため、heatmap path 自体を予測値にせず、exp157/158 selector の confidence feature として add-only する。

### 作成

- `.steering/20260703-exp184-cnn-sdf-mtp-heatmap-path-features-on-exp158/` を作成。
- `experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/` を exp183 pattern からコピーして作成。
- 実装本体を `cnn_sdf_mtp_heatmap_path_features_on_exp158.py` として作成。
- `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_train.py` / `_inference.py` を Jupytext percent source として更新し、`.ipynb` に変換。

### 実装内容

- exp157 と同じ 8候補、exp099 multiobs feature、exp072 dense enrichment、GroupKFold 5 folds、LightGBM 3 configs、exp158 Viterbi grid を維持。
- exp182 `base_real_w128_b64_fullfold` validation predictions から、topK path center、topK score margin、entropy、topK spread、path-step stats、prior center relation、PF/Beam/dense candidate との差分を生成。
- exp182 `base_shuffled_w128_b64_fullfold` と `base_no_gr_w128_b64_fullfold` は real-vs-control gap feature のみに使用。
- exp182 validation predictions は sparse sample なので、well 内 `row_center` による線形補間で exp158 selector row に展開。
- `pred_top*_abs_error`、`top*_within10`、true TVT、target_in_grid、oracle label、true-error rank は feature source として読み込まない。
- candidate-long feature として heatmap topK path と候補 TVT の差分、topK min abs distance、confidence x PF/Beam/dense family interaction を追加。
- heatmap confidence bucket、nearest sparse sample distance bucket、exp115 hidden-like subgroup を diagnostic として保存。

### GPU / booster cost guard

- 実行対象: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- active selector variant: 1
- LightGBM configs: 3
- folds: 5
- planned boosters: 15
- parent/control retraining: なし
- GPU: なし。Kaggle CPU train。
- direct replacement / inference port / submit: なし。

### 再現性

- exp184 自体の feature generation は deterministic。heatmap interpolation に乱数なし。
- upstream exp182 は PyTorch CUDA diagnostic artifact のため、deterministic submission anchor とは扱わない。
- LightGBM seed、GroupKFold seed、candidate-long row subsample seed は固定。
- gzip input は decompressed SHA を summary に記録する方針。
- submission は生成しない。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/cnn_sdf_mtp_heatmap_path_features_on_exp158.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_train.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_inference.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/settings.py
```

結果: pass。

```bash
.venv/bin/ruff check \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/cnn_sdf_mtp_heatmap_path_features_on_exp158.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_train.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_inference.py \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/settings.py \
  --select F821
```

結果: pass。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158_inference.py
```

結果: pass。

```bash
PYTHONPATH=experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 \
  .venv/bin/python -c "from settings import load_config; from cnn_sdf_mtp_heatmap_path_features_on_exp158 import candidate_specs_from_config, build_required_columns, read_heatmap_predictions; c=load_config(); specs=candidate_specs_from_config(c); print('candidates', [s.name for s in specs]); print('required_columns', len(build_required_columns(c, specs))); frame, meta = read_heatmap_predictions(c); print('heatmap_rows', len(frame), 'wells', frame['well'].nunique()); print('run_specs', sorted(meta['run_spec_counts'].items()))"
```

結果:

- candidates: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `tvt_dense`, `tvt_densew`, `tvt_dense50`
- required columns: 42
- heatmap rows: 32,466
- heatmap wells: 773
- run specs: `base_real_w128_b64_fullfold`, `base_shuffled_w128_b64_fullfold`, `base_no_gr_w128_b64_fullfold` が各 10,822 rows

```bash
make validate-exp EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158
```

結果: `experiment validation passed (strict)`。

追加修正後に再実行:

- `py_compile`: pass
- `ruff --select F821`: pass
- heatmap feature dummy check: `heatmap_cols=126`、`configured_missing=[]`、`source_valid=1.0`
- `make validate-exp EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`: pass

`git status --short` は、この workspace が Git repo として見えていないため `fatal: not a git repository` で確認できなかった。

### ローカル未確認事項

2,000 行の feature assembly smoke は、ローカルに exp072 dense train cache 本体がないため `FileNotFoundError` で停止した。これは exp183 と同じ upstream dependency で、Kaggle train では `kentookumura/exp072-exp063-full-replay-feature-cache-train` kernel source から解決する想定。

不足ファイル:

- `experiments/exp072_exp063_full_replay_feature_cache/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`

### 次アクション

- `make prepare-kaggle-notebooks EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp184-hmpf-train --title 'exp184 hmpf train' --run-on-push --strict"`
- Kaggle train 後、exp158 continuity RMSE 10.789163253、path switch、worst-well、heatmap confidence bucket、exp115 hidden-like subgroup、feature importance を確認する。

## 2026-07-04 Kaggle train 実行

### 実行前ガード

- 実行対象: Kaggle train notebook `kentookumura/exp184-hmpf-train`
- active variant: 1 add-only heatmap-path selector variant
- LightGBM configs: 3
- folds: 5
- planned boosters: 15
- continuity Viterbi variants: 180
- control / parent retraining: none
- GPU: disabled
- inference / direct replacement / submit: out of scope

### Push メモ

- 初回の長い kernel id `kentookumura/exp184-cnn-sdf-mtp-heatmap-path-features-on-exp158-train` は Kaggle API `400 Bad Request` で reject されたため、kernel 名のみ `kentookumura/exp184-hmpf-train` に短縮して再実行する。
- `make push-kaggle-train EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`: success。Kaggle kernel version 1 pushed。
- URL: https://www.kaggle.com/code/kentookumura/exp184-hmpf-train
- push 後の `kaggle kernels status kentookumura/exp184-hmpf-train`: `KernelWorkerStatus.RUNNING`
- `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp184-hmpf-train`: 5 分追跡したが stdout/stderr はまだ空。再確認時点でも status は `RUNNING`。

## 2026-07-04 Kaggle train v1 失敗確認と retry

### v1 status

- `kaggle kernels status kentookumura/exp184-hmpf-train`: `KernelWorkerStatus.ERROR`
- `kaggle kernels logs kentookumura/exp184-hmpf-train`: fold0 multiclass は best iteration 124 で完了。その後、`DeadKernelError: Kernel died`。
- `kaggle kernels output kentookumura/exp184-hmpf-train -p experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/kaggle/output/train_v1`: 取得成功。部分生成物は fold0 の `lgb_multiclass` model のみで、metrics / summary は未生成。

### 原因仮説

- exp184 は heatmap path feature 追加で row feature が増えている一方、long-format binary/error ranker が full valid long frame を保持し、train sample も 650k rows/fold だった。
- fold0 multiclass 後、binary/error 用 long frame 構築または学習前後でメモリが膨らみ、Kaggle kernel が kill された可能性が高い。

### v2 修正

- exp183 の既存メモリ対策を移植。
- `ranker.long_models.max_train_rows_per_fold`: `650000` -> `120000`
- `ranker.long_models.max_valid_rows_per_fold`: `120000` を追加。
- `ranker.long_models.predict_chunk_rows`: `50000` を追加。
- binary/error の early stopping eval は bounded sample、full valid の OOF score は chunk prediction で生成する。
- fold ごとに long frame / numpy matrix / predictions を明示的に `del` し、`gc.collect()` する。

### v2 検証

- `py_compile`: pass
- `ruff --select F821`: pass
- `make validate-exp EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`: pass
- `make prepare-kaggle-notebooks EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp184-hmpf-train --title 'exp184 hmpf train' --run-on-push --strict"`: pass
- `make push-kaggle-train EXP=exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`: success。Kaggle kernel version 2 pushed。
- push 後の `kaggle kernels status kentookumura/exp184-hmpf-train`: `KernelWorkerStatus.RUNNING`
- `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp184-hmpf-train`: 5 分追跡したが stdout/stderr はまだ空。再確認時点でも status は `RUNNING`。

## 2026-07-04 Kaggle train v2 完了

### 完了確認

- `kaggle kernels status kentookumura/exp184-hmpf-train`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp184-hmpf-train`: summary と artifact list を確認。
- `kaggle kernels output kentookumura/exp184-hmpf-train -p experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/kaggle/output/train_v2`: 取得成功。

### 結果

- status: `completed_train_side_audit`
- runtime: `33550.90019130707` sec
- rows / wells: `3,783,989` / `773`
- feature count: `223`
- best non-oracle: `viterbi_sw200_bias000_jw000_jf025_d0150_std999999_md0000_seg012`
- best RMSE: `10.560650324533297`
- MAE: `6.329187985951713`
- within10: `0.7970564925003746`
- path switches: `5713` (`1.5097824015873198` / 1000 rows)
- delta vs exp158 continuity `10.789163253`: `-0.2285129284667029`
- delta vs exp157 row-wise `10.79579983712686`: `-0.23514951259356387`
- delta vs `likpf_mean` `11.594897672217703`: `-1.0342473476844063`

### Diagnostics

- selection distribution: `likpf_mean` 42.25%、`pf_ancc` 38.26%、dense family 15.13%、`beam_mean` 4.33%。
- worst well: `86454a6f`, RMSE `57.960134`, within10 `0.046208`。
- heatmap sparse distance bucket: q1 RMSE `7.042431`, q4 RMSE `14.058409`。
- exp115 spatial valid RMSE `12.696140`、typewell purged valid RMSE `12.629861`。
- `hmpf_far_from_sparse_sample_gt512` RMSE `13.029168`。
- heatmap candidate-distance features are used by LightGBM; top heatmap features include `hmpf_real_top10_mean_minus_candidate_abs`, `hmpf_real_top1_minus_candidate_abs`, and `hmpf_real_top5_mean_minus_candidate_abs`.

### 判断

- train-side では exp158 から明確に改善し、heatmap path add-only selector feature は supported。
- ただし inference / submit は未実施。raw-test heatmap generation、sparse interpolation coverage、feature schema parity、fallback behavior、hidden-like subgroup stress を同じ exp184 内で確認してから判断する。
