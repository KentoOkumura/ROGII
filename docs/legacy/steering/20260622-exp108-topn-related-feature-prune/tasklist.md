# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `experiments/exp108_topn_related_feature_prune/` を exp098 から作成した。
- `topn_related_feature_prune.py` を実装した。
- `exp098_full_260`、`top1_related_pruned_260`、`top2_related_pruned_260`、`top3_related_pruned_260`、`non_candidate_context_plus_topn_related` の config を追加した。
- GPU 節約のため、既存 exp098 の rank-slot distribution と feature importance から top3 を採用し、active variant を `top3_related_pruned_260` のみにした。
- inference notebook を train-side audit only の guard に変更した。
- `py_compile`、notebook JSON validation、`make validate-exp`、ruff を通した。
- Kaggle train package を作成した。
- Kaggle train v1 を実行し、`KernelWorkerStatus.COMPLETE` を確認した。
- output を `experiments/exp108_topn_related_feature_prune/kaggle/output/train_v1` に取得した。
- feature schema / prediction / model manifest SHA を記録した。
- OOF、worst-well、distance bucket、feature importance を exp073 / exp077 / exp098 / exp105 / exp092 と比較した。
- OOF 悪化のため `completed_train_side_rejected` とし、inference / submit しない判断を記録した。
