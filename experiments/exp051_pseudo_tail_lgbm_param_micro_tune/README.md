# exp051_pseudo_tail_lgbm_param_micro_tune

## 状態

- ルート: MLモデル
- 状態: completed
- CV: 12.634392
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-10
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`

## 仮説

`exp026` の pseudo-tail + distance-balanced LightGBM 構成は強いが、`num_leaves`、`min_child_samples`、subsample/colsample、正則化、row cap の狭い調整で fold 4 や遠距離 bucket を少し改善できる可能性がある。

## 検証方針

主評価は従来の well-level GroupKFold を維持する。raw LightGBM CV と exp025-selected fixed bucket-shrink 後 CV を variant 別に保存し、`exp026` fixed bucket-shrink 12.870780 を安定して上回る候補だけ次段の補助検証へ進める。

## 所見

Kaggle train version 1 で full CV 完了。best は `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params` 12.634392。推論 notebook と提出処理はこの実験では使わない。

## 参照ファイル

- 設定: `config.yaml`
- セッションノート: `SESSION_NOTES.md`
- 結果: `result.md`
- メトリクス: `metrics.json`
- 学習 notebook: `exp051_pseudo_tail_lgbm_param_micro_tune_train.ipynb`
- 推論 notebook: `exp051_pseudo_tail_lgbm_param_micro_tune_inference.ipynb`
