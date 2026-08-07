# exp054_pseudo_tail_seed_bagging_inference_submit

## 状態

- ルート: MLモデル
- 状態: completed
- CV: 12.633797
- Public LB: 11.856
- Private LB: -
- Submit ID: 53526321
- 作成日: 2026-06-10
- 親実験: `exp053_pseudo_tail_seed_bagging`

## 仮説

`exp053` の 3-seed bagging は exp051 best と CV では実質同等だが、seed 変更が Public LB へどう出るかを早めに確認する価値がある。

## 検証方針

`exp052` の inference flow を維持し、final residual model だけ 3-seed 平均へ変更する。Kaggle output の `submission.csv` を sample submission と照合し、exp052 submission との差分を確認してから code submit する。

## 所見

Kaggle inference version 1 と code submit を完了。ref `53526321` の Public LB は 11.856。submit-check は PASS、exp052 submission との差分 RMSE は 1.113177。pseudo-tail 自前系の Public LB 基準は更新するが、ML route 全体基準 exp039 11.740 には届かない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 推論 notebook: `exp054_pseudo_tail_seed_bagging_inference_submit_inference.ipynb`
