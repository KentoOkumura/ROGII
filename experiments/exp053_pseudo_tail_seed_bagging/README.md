# exp053_pseudo_tail_seed_bagging

## 状態

- ルート: MLモデル
- 状態: completed
- CV: 12.633797
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`

## 仮説

`exp051` の LightGBM capacity pseudo-tail model は中距離から遠距離で強くなった一方、single seed の sampling / model variance が残っている可能性がある。pseudo-tail sampling seed と LightGBM seed を 3 本に増やして raw prediction を平均すれば、構造を変えずに fold / bucket の分散を下げられる可能性がある。

## 検証方針

主評価は従来の well-level GroupKFold を維持する。`lgbm_capacity_single_seed_control` と `lgbm_capacity_seed_bag3` の raw / fixed bucket-shrink RMSE を保存し、same-run control と `exp051` best 12.634392 を比較する。

## 所見

Kaggle train version 1 で full CV 完了。best は `lgbm_capacity_seed_bag3_exp014_bucket_shrink_params` 12.633797。same-run single seed control fixed 12.734551 からは -0.100754 改善したが、直近基準の exp051 best 12.634392 との差は -0.000595 とごく小さい。推論 port は高優先にはしない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp053_pseudo_tail_seed_bagging_train.ipynb`
- 推論 notebook: `exp053_pseudo_tail_seed_bagging_inference.ipynb`
