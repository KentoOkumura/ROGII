# exp108_topn_related_feature_prune セッションノート

## 2026-06-22 実装

### コマンド

- `make new-steering EXP=exp108_topn_related_feature_prune`
- `make new-exp EXP=exp108_topn_related_feature_prune SOURCE=experiments/exp098_selector_rank_slot_features_on_exp073`
- `.venv/bin/python -m py_compile experiments/exp108_topn_related_feature_prune/topn_related_feature_prune.py experiments/exp108_topn_related_feature_prune/settings.py`
- `.venv/bin/python -m json.tool experiments/exp108_topn_related_feature_prune/exp108_topn_related_feature_prune_train.ipynb`
- `.venv/bin/python -m json.tool experiments/exp108_topn_related_feature_prune/exp108_topn_related_feature_prune_inference.ipynb`
- `make validate-exp EXP=exp108_topn_related_feature_prune`
- `.venv/bin/ruff check experiments/exp108_topn_related_feature_prune/topn_related_feature_prune.py experiments/exp108_topn_related_feature_prune/settings.py`
- `make prepare-kaggle-notebooks EXP=exp108_topn_related_feature_prune EXTRA_ARGS="--notebook train --run-on-push --strict"`
- `make prepare-kaggle-notebooks EXP=exp108_topn_related_feature_prune EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp108-topn-related-feature-prune-train --title 'exp108 topn related feature prune train' --run-on-push --strict"`
- `make push-kaggle-train EXP=exp108_topn_related_feature_prune`
- `kaggle kernels pull kentookumura/exp108-topn-related-feature-prune-train -p /tmp/kaggle-pull/exp108-topn-related-feature-prune-train -m`
- `kaggle kernels logs kentookumura/exp108-topn-related-feature-prune-train`
- `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp108-topn-related-feature-prune-train`
- `kaggle kernels status kentookumura/exp108-topn-related-feature-prune-train`
- `kaggle kernels output kentookumura/exp108-topn-related-feature-prune-train -p experiments/exp108_topn_related_feature_prune/kaggle/output/train_v1`

### 実装メモ

- `docs/legacy/steering/20260622-exp108-topn-related-feature-prune/` を作成。
- exp098 を親として実験ディレクトリを作成。
- 実装ファイルを `topn_related_feature_prune.py` にリネームし、出力 prefix を exp108 に変更。
- exp098 の full rank-slot feature generation を維持しつつ、base 196 features を `base_196_all`、`base_196_candidate_family`、`base_196_non_candidate_context`、`base_196_topn_core_candidate_family` に分類する処理を追加。
- rank-slot features を `rank_slot_top1_related`、`rank_slot_top2_related`、`rank_slot_top3_related`、`rank_slot_source_flags`、`rank_slot_global_candidate_stats`、`rank_slot_pairwise_disagreement` に分類する処理を追加。
- `feature_columns_for_variant()` を base feature group と rank-slot feature group の両方を扱えるように変更。
- `config.yaml` は control `exp098_full_260` と prune variants 4 本を定義しつつ、GPU 節約のため active variant を `top3_related_pruned_260` だけに変更。
- top3 固定の根拠は exp098 の既存結果。rank3 は `pf_ancc` 41.26%、`beam_mean` 52.26% で、`rank3_u_curvature` / `rank3_u_slope` / `rank3_u_resid_mad` が特徴量重要度上位に入っている。`sc_ens` / `hyb` は top3 でもほぼ選ばれないため top4/top5 は使わない。
- inference notebook は train-side audit only の guard に変更。
- base feature group smoke で列数を確認: all=196、candidate_family=44、non_candidate_context=152、topn_core_candidate_family=7。
- `py_compile`、notebook JSON validation、`make validate-exp`、ruff は pass。
- Kaggle train package は `experiments/exp108_topn_related_feature_prune/kaggle/train` に生成済み。
- train kernel id は `kentookumura/exp108-topn-related-feature-prune-train`。
- 初回 push は Kaggle SaveKernel 400 `Your kernel title does not resolve to the specified id` で失敗したため、kernel id に解決される短い title `exp108 topn related feature prune train` で package を再生成した。同じ canonical kernel id を維持した。
- 再 push は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp108-topn-related-feature-prune-train
- `kaggle kernels pull ... -m` は成功し、metadata / source の存在確認済み。
- 通常 logs と 3分 follow logs は空。Kaggle CLI の session log がまだ返っていない状態として扱い、別 slug への再 push はしていない。
- 初回確認時の `kaggle kernels status kentookumura/exp108-topn-related-feature-prune-train` は `KernelWorkerStatus.RUNNING`。

## 2026-06-22 Kaggle train v1 完了

### コマンド

- `kaggle kernels status kentookumura/exp108-topn-related-feature-prune-train`
- `kaggle kernels logs kentookumura/exp108-topn-related-feature-prune-train`
- `kaggle kernels output kentookumura/exp108-topn-related-feature-prune-train -p experiments/exp108_topn_related_feature_prune/kaggle/output/train_v1`

### 結果

- status: `KernelWorkerStatus.COMPLETE`
- runtime: 8775.76 sec
- rows / wells: 3,783,989 / 773
- active variant: `top3_related_pruned_260`
- features: 195 (`base_196_non_candidate_context` 152 + `base_196_topn_core_candidate_family` 7 + `rank_slot_top3_related` 36)
- pooled OOF:
  - `lgb2`: 9.479370656
  - `lgb1`: 9.491034034
  - `lgb_mean`: 9.529005954
  - `lgb0`: 9.798771537
- 比較:
  - vs exp073 raw anchor: -0.047004094
  - vs exp077 policy: +0.008855855
  - vs exp098 best: +0.121219603
  - vs exp098 lgb_mean: +0.101557966
  - vs exp105 best: +0.038267495
  - vs exp092 best: +0.156890760

### 判断

top3 関連 feature への静的 prune は rejected。exp098 full 260 features から context / disagreement / candidate-family signal を削ると、rank-slot U-shape features が残っていても OOF が悪化した。

inference / submit は行わない。exp098 full rank-slot を比較基準として維持し、次は prune ではなく exp092 への小さな add-only merge、または candidate-generation / likelihood 側の改善を優先する。

### SHA

- model manifest: `e773850dd0f9eac74a416d1936cdf9a9b2a511dda094083952919e2da253b88b`
- predictions gzip raw: `b16061aab916298e141a0482d2cfcee8cb06d5de1f6ab91592a5b9978cbc302b`
- predictions decompressed: `993fbf4e48a4e612e8b3a26d3d26fccc2d29d412e911a13098eeded4a11844ef`
- feature schema: `1633b2628df46a92a23597f6807992f64b3fe6bf5dd42baa33bbb80ba54821e7`
- source cache: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- feature schema source: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- summary source: `133f9be7a6bcf8606e18b7d41f4d24d84e1d8e0f128660717b21fea4fad46b7f`
