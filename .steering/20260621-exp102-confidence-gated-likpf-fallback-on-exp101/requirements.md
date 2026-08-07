# 要件

## 依頼

`confidence_gated_likpf_fallback_on_exp101` を実装する。

exp101 の row-wise supervised selector は `likpf_mean` 単体を超えなかったが、`pf_ancc` / `beam_mean` への切替が高信頼な行だけに限定できれば、oracle headroom の一部を低リスクに拾える可能性がある。exp101 の保存済み output / model を使い、posthoc に conservative gate を監査する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- default prediction は常に `likpf_mean`。
- high-confidence 条件を満たす行だけ、exp101 `lgb_candidate_error_ranker` の selected candidate へ切り替える。
- 切替候補は `pf_ancc` / `beam_mean` を主対象にし、低 switch-rate の小 grid に限定する。
- 評価は train-side OOF audit のみ。提出候補化や inference port はこの実装では行わない。

## 受け入れ基準

- exp102 の `config.yaml`、train notebook、補助 `.py`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が実験内容と一致している。
- exp101 model manifest / booster と exp099 v2 train feature cache を読み、OOF fold-safe に per-candidate predicted error / probability margin を復元する。
- `likpf_mean` baseline、exp101 row-wise best、confidence-gated variants、oracle を同じ行集合で比較し、RMSE、within10、switch rate、selection distribution、path switch、bucket / worst-well を出力する。
- deterministic anchor として扱わず、入力 SHA、model manifest SHA、prediction SHA を train-side audit の証拠として記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
