# 要件

## 依頼

`gr_shape_descriptor_verifier_on_candidate_selector` の実装に進む。

## 制約

- Route: `pf_beam`
- `likpf_mean` を default path とし、descriptor score 単独 argmax、softmax weighted average、direct candidate replacement はしない。
- exp131 の GR shape descriptor signal は verifier / veto / confidence 補助としてのみ使う。
- valid/test true TVT、oracle candidate、true error を gate 条件や feature source に漏らさない。
- row-wise switch を増やす設定は rejected とし、switch rate と path switch を必ず記録する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/` に train-side audit 実装がある。
- exp101 saved booster と exp099 v2 cache から OOF score surface を復元できる。
- raw train GR と visible `TVT_input` prefix から exp131 相当の descriptor score を再計算できる。
- `likpf_mean_single`、`exp101_error_ranker_rowwise`、`oracle`、descriptor verifier variants を同一 surface で比較する。
- overall RMSE、within10、switch rate、selection distribution、by-well、bucket metrics、descriptor score summary、prediction SHA を保存する。
- train notebook が設定、入力確認、実行、生成物 preview をセル単位で追える。
- inference notebook は train-side audit only とし、`submission.csv` を生成しない。
