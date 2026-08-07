# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `.steering/20260627-exp147-exp092-exp098-rank-slot-replacement-only/` を作成した。
- `experiments/exp147_exp092_exp098_rank_slot_replacement_only/` を exp139 から作成した。
- 再現性設計を `design.md` に記入した。
- replacement-only の drop columns / replacement columns を `config.yaml` に明記した。
- 補助実装に `drop_columns` 対応を追加した。
- py_compile、notebook JSON、`make validate-exp`、ruff、synthetic feature-column smoke test を通した。
- Kaggle train push 前に metadata と bootstrap 内 config の整合を確認した。
- Kaggle train push 前に active variant / config / fold / booster 数を再確認し、`SESSION_NOTES.md` に追記した。
- Kaggle train v1 を `kentookumura/exp147-rank-slot-replacement-train` で実行完了した。
- Kaggle output を取得し、metrics、bucket、by-well、feature schema、model manifest、prediction SHA、artifact SHA を記録した。
- exp092 / exp139 と比較し、train-side rejected / no submit と判断した。
