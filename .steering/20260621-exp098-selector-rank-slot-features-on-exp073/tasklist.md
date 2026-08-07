# タスクリスト

## TODO

- Kaggle train を push して full OOF を取得する。
- metrics、by-well、bucket、feature importance、rank-slot source distribution を確認する。
- exp073 / exp077 / exp092 との比較を result と experiment_summary に追記する。
- 改善した場合も inference port 前に worst-well regression guard、near rows、path continuity、raw-test feature parity を確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering requirements / design / tasklist を作成した。
- `exp098_selector_rank_slot_features_on_exp073` 実験ディレクトリを作成した。
- config に route、lineage、rank-slot candidates、単一 selected variant、LightGBM modes、Kaggle source を記録した。
- `selector_rank_slot_features_on_exp073.py` に rank-slot structured feature generation と exp073 surface LightGBM ablation runner を実装した。
- train notebook を設定表示、入力 plan、学習、metrics 保存の構成へ更新した。
- inference notebook を未選択として明示的に停止する構成へ更新した。
- Python 構文検査と notebook JSON 検査を通した。
- `make validate-exp EXP=exp098_selector_rank_slot_features_on_exp073` を通した。
- Kaggle train package を strict mode で作成した。
