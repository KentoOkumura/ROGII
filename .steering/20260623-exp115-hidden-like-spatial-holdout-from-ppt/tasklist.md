# タスクリスト

## TODO

- Kaggle train notebook を prepare して push する。
- Kaggle output を取得し、`summary.json` / `holdout_wells.csv` を確認する。
- Kaggle output 取得後に `SESSION_NOTES.md`、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を Kaggle 実行結果で更新する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs を作成した。
- 再現性設計を `design.md` に記入した。
- stochastic 処理なし、global RNG / thread scheduling 依存なしの設計にした。
- `config.yaml` を exp115 の holdout audit 用に更新した。
- `hidden_like_spatial_holdout_from_ppt.py` を追加した。
- train notebook を Kaggle 実行用の監査 notebook に更新した。
- inference notebook は no-submission 明示に更新した。
- `make validate-exp EXP=exp115_hidden_like_spatial_holdout_from_ppt` が通った。
- local smoke で PPT red component 45 件、holdout 200/200 wells を生成した。
- `ruff check` が通った。
- `make prepare-kaggle-notebooks EXP=exp115_hidden_like_spatial_holdout_from_ppt EXTRA_ARGS="--notebook train --strict"` が通った。
- `SESSION_NOTES.md`、`result.md`、`README.md`、`KAGGLE_DIRECTION.md` を local smoke 結果で更新した。
