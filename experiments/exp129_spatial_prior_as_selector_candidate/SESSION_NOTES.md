# exp129_spatial_prior_as_selector_candidate セッションノート

## 2026-06-25 実装

ユーザー依頼により `spatial_prior_as_selector_candidate` を実装する。

### 狙い

exp118 では spatial prior を exp092 に小さく hard correction すると RMSE 改善は `9.322480 -> 9.321625` と微小だった。一方で、spatial prior は「いつ信用するか」を判定できれば候補 path として使える可能性がある。

今回は exp099/101 系の fixed candidate surface に exp114 の fold-safe spatial prior TVT を追加し、raw replacement ではなく candidate selector の選択肢として headroom を診断する。

### 実装内容

- `docs/legacy/steering/20260625-exp129-spatial-prior-as-selector-candidate/` を作成。
- `experiments/exp129_spatial_prior_as_selector_candidate/` を exp101 から派生作成。
- `spatial_prior_as_selector_candidate.py` を追加。
  - exp099 v2 train feature cache を読む。
  - exp114 v1 OOF spatial prior artifact を `id, well` で結合する。
  - base 5候補 (`pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`) に `xy_plus_trajectory_shape_k8_prior_tvt` と `xy_only_k8_prior_tvt` を追加する。
  - base oracle / spatial-only oracle / expanded oracle、candidate metrics、true-error topK を保存する。
  - candidate-long predicted-error LightGBM と Viterbi switch penalty selector を OOF で比較する。
  - metrics、OOF selected prediction、selection distribution、by-well path switch、bucket metrics、feature importance、model manifest、summary JSON を保存する。
- train notebook は設定、入力確認、候補計画、実行、結果 preview をセル単位で追える構成にする。
- inference notebook は train-side audit only と明記する。

### 再現性

- 新規 PF/Beam / spatial prior 生成はしない。
- exp099 cache と exp114 OOF artifact は gzip decompressed SHA を主証拠として記録する。
- LightGBM と row subsample は seed 固定だが、deterministic submission anchor とは扱わない。
- Kaggle CPU / internet off / kernel sources は exp099 train と exp114 train。

### 次

- Kaggle train v1 の完了後に output を取得し、結果を `result.md` / `metrics.json` / `experiment_summary.md` に反映する。

### ローカル検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp129_spatial_prior_as_selector_candidate/spatial_prior_as_selector_candidate.py \
  experiments/exp129_spatial_prior_as_selector_candidate/settings.py
.venv/bin/ruff check experiments/exp129_spatial_prior_as_selector_candidate
make validate-exp EXP=exp129_spatial_prior_as_selector_candidate
```

成功。

`--skip-models` の debug smoke で、exp099 cache と exp114 OOF artifact の結合、base/spatial/expanded oracle、candidate metrics、true-error topK、生成物保存を確認した。

```bash
EXPERIMENT_ALLOW_LOCAL=1 .venv/bin/python \
  experiments/exp129_spatial_prior_as_selector_candidate/spatial_prior_as_selector_candidate.py \
  --output-dir /tmp/exp129_smoke \
  --max-rows 2000 \
  --skip-models
```

結果:

- rows: 2,000
- wells: 1
- feature_count: 76
- best debug metric: `oracle_expanded` RMSE 3.814549
- expanded oracle delta vs base oracle: -0.001230

debug smoke では full gzip SHA 計算を省略する。正式 Kaggle train は `selector.max_rows=null` のため exp099 / exp114 の raw SHA と decompressed SHA を記録する。

### Kaggle train v1

```bash
make prepare-kaggle-notebooks EXP=exp129_spatial_prior_as_selector_candidate \
  EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp129-spatial-selector-train --title 'exp129 spatial selector train' --run-on-push --strict"
make push-kaggle-train EXP=exp129_spatial_prior_as_selector_candidate
```

push 成功。

- Kernel: `kentookumura/exp129-spatial-selector-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp129-spatial-selector-train`
- version: 1
- CPU runtime (`enable_gpu=false`)
- internet: false
- kernel sources:
  - `kentookumura/exp099-pf-multiobs-likelihood-train`
  - `kentookumura/exp114-spatial-neighbor-prior-signal-audit-train`

`timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp129-spatial-selector-train` はログ空のまま timeout。通常 `logs` も warning 以外は空。`kaggle kernels status kentookumura/exp129-spatial-selector-train` は `KernelWorkerStatus.RUNNING`。

## 2026-06-26 Kaggle train v1 完了

### 状態確認と output 取得

```bash
kaggle kernels status kentookumura/exp129-spatial-selector-train
kaggle kernels logs kentookumura/exp129-spatial-selector-train
kaggle kernels output kentookumura/exp129-spatial-selector-train \
  -p experiments/exp129_spatial_prior_as_selector_candidate/kaggle/output/train_v1
```

status は `KernelWorkerStatus.COMPLETE`。output 取得済み。

### 結果

rows 3,783,989 / wells 773 / runtime 3,797.765 秒。

| variant | mode | RMSE | MAE | within10 | oracle acc |
| --- | --- | ---: | ---: | ---: | ---: |
| `oracle_expanded` | oracle | 6.709127 | 3.138080 | 0.929853 | 1.000000 |
| `oracle_base_only` | oracle | 7.434030 | 3.745228 | 0.906525 | 0.784658 |
| `likpf_mean_single` | baseline | 11.594898 | 7.067633 | 0.772807 | 0.321356 |
| `lgb_error_ranker_rowwise` | oof | 13.793157 | 7.187177 | 0.769536 | 0.312807 |
| `lgb_error_ranker_viterbi_p0p25` | oof_viterbi | 13.793777 | 7.185637 | 0.769577 | 0.313877 |
| `oracle_spatial_only` | oracle | 14.353528 | 9.629028 | 0.651661 | 0.215755 |

expanded oracle は base oracle から RMSE -0.724903 改善。spatial 候補の oracle top1 は合計 21.53% で、`xy_plus_trajectory_shape_k8_prior_tvt` 10.8478%、`xy_only_k8_prior_tvt` 10.6863%。true-error topK に spatial が入る割合は top1 21.53%、top2 41.26%、top3 67.78%、top5 95.74%。

一方で best OOF selector は `lgb_error_ranker_rowwise` RMSE 13.793157 で、`likpf_mean_single` RMSE 11.594898 より +2.198259 悪化。spatial selection rate は 10.97% あるが、信用判定に失敗している。Viterbi は switch を減らすものの RMSE は改善しない。

### 生成物

- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_summary.json`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_candidate_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_selection_distribution.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_oof_selected_predictions.csv.gz`
- `kaggle/output/train_v1/artifacts/exp129_spatial_prior_as_selector_candidate_model_manifest.json`

### SHA

- exp099 input raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 input decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp099 schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- exp114 input raw SHA: `7a328efd941b4acce476622d3e65e775c65bc9a385c600cdfed9efe3f0d75aa0`
- exp114 input decompressed SHA: `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`
- exp114 summary SHA: `5ee7a7af6b05cf523a1ce1353e389c01a9b50b12c8a77441e4bf7199a6ab1e94`
- metrics SHA: `788796940937180127a8d17ec520a5e5d97ace96877bc1a72e8aa7104694d17f`
- OOF predictions decompressed SHA: `5c1a0b66154ed7e2d58e38b804c3953614547e87293dded9d980a93e96495617`
- model manifest SHA: `65030ac43f78cb5e82d9190753a5cc9d8c1b28c15d20929cfed6e501660c778a`

### 解釈

`spatial_prior_as_selector_candidate` は direct selector として不採用。spatial prior は oracle headroom を持つが、exp099/101 系の predicted-error selector と Viterbi smoothing では `likpf_mean` を超えない。今後は候補 path 選択ではなく、`spatial_neighbor_prior_ml_features_on_exp092` の confidence / add-only feature として扱う。
