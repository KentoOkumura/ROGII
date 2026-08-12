# 要件

## 依頼

`pf_confidence_residual_clip` を exp077 の付随変更ではなく、独立した backlog 実験として実行する。exp077 は `longtail_likpf_tiny_gate_w006` の submitted anchor として保持し、exp084 では `pf_confidence_residual_clip_q995` の Kaggle inference output を取得する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新しい教師あり学習は行わない。
- exp073 saved booster と raw test PF/Beam/likelihood-PF replay を使う。
- exp077 の anchor config は `longtail_likpf_tiny_gate_w006` に戻す。

## 受け入れ基準

- Kaggle inference notebook が `pf_confidence_residual_clip_q995` で実行される。
- `submission.csv` が sample submission と互換である。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
