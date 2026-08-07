# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある
`candidate_distribution_router` を実験として実装する。

## 仮説

PF/beam と DTW/DWT は add-only features では悪化したが、候補予測としては
row distance や候補間 disagreement が良い局所条件だけで使えば、raw
`lightgbm_no_gr` の誤差を補える可能性がある。

## 制約

- 新規モデル学習はしない。既存 OOF artifact の監査に限定する。
- 評価面は `exp013` の `lightgbm_no_gr` OOF と同じ evaluation-zone rows に揃える。
- 任意の PF/beam row OOF がローカルにない場合は再生成せず、スキップして実行可能にする。
- Same-OOF router/oracle score は診断値として扱い、clean CV とは分ける。
- 採用判断は leave-one-original-fold-out と stable well-hash holdout selection を優先する。

## 受け入れ基準

- `exp018_candidate_distribution_router` が作成され、設定、entrypoint、監査スクリプトが揃っている。
- `last_anchor`、raw LightGBM、HGB control、DTW/DWT、任意 PF/beam 候補を扱える。
- fixed candidate、blend、distance router、disagreement damping、bucket oracle を比較できる。
- router metrics、selection audit、bucket summary、summary JSON を artifact として出力できる。
- `validate_experiment.py` と静的チェックが通る。
