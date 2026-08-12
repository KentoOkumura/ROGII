# タスクリスト

## TODO

- 改善候補がある場合だけ exp044 補助検証または inference port を別実験として切る。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- `exp051_pseudo_tail_lgbm_param_micro_tune` を生成した。
- LightGBM micro tune 候補を `config.yaml` に追加した。
- variant 別 `model_params` と row cap override を training loop に実装した。
- `ruff check`、`py_compile`、`validate_experiment` を通した。
- Kaggle train notebook package を作成した。
- Kaggle train version 1 を実行し、metrics と CSV 生成物を取得した。
- 主評価 GroupKFold の raw / fixed bucket-shrink CV、fold、distance bucket を記録した。
- `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params` を選択候補として記録した。
