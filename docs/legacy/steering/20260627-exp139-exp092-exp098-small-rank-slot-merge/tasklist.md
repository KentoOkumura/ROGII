# タスクリスト

## TODO

- Kaggle train package を作成し、metadata と bootstrap manifest を確認する。
- Kaggle train を実行し、OOF / worst-well / bucket / path continuity / feature importance を exp092、exp098、exp107、exp108 と比較する。
- output 取得後に feature content SHA、prediction SHA、model SHA、decompressed gzip SHA を記録する。
- train-side guard が通った場合だけ inference notebook を package 化する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/legacy/steering/20260627-exp139-exp092-exp098-small-rank-slot-merge/` を作成した。
- `experiments/exp139_exp092_exp098_small_rank_slot_merge/` を exp092 から作成した。
- exp098 の target-free rank-slot feature generator を exp139 実装へ移植した。
- exp092 U-projection feature surface と small rank-slot add-only columns を結合する train/inference runner を実装した。
- `config.yaml` を exp139 の仮説、small merge columns、比較対象、再現性方針に更新した。
- train / inference notebook 名と参照関数を exp139 に更新した。
- README、SESSION_NOTES、result、metrics を実装済み・未実行状態に更新した。
