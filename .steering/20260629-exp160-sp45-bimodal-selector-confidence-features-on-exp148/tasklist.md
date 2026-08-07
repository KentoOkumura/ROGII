# タスクリスト

## TODO

- Kaggle train push 前に 15 boosters / control 再学習なしを再確認する。
- Kaggle train 完了後に OOF、near `000_050`、`1000_plus`、worst-well、common PF+ML worst wells、feature importance、raw-test/current-test parity を記録する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs 作成。
- exp148 から exp160 を scaffold。
- `sp45_bimodal_selector_confidence_features_on_exp148.py` を実装。
- train / inference notebook を exp160 用に更新。
- `SESSION_NOTES.md`、`README.md`、`result.md`、`metrics.json` を未実行状態に更新。
- Python compile と notebook JSON validation を実行。
- `ruff check` を通過。
- `make validate-exp EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` を通過。
- train / inference notebook を `prepare-kaggle-notebooks --strict` で package 化。
- long canonical train kernel id は Kaggle `SaveKernel` 400 で失敗したため、同じ exp160 のまま shorter id/title へ切り替える判断を記録。
