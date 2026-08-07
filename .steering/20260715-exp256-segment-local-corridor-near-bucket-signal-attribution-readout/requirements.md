# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ
`segment_local_corridor_near_bucket_signal_attribution_readout` を実装する。
exp250 の 0--100 ft で見えた pooled real AUC 約 0.82 が、GR topology 固有の
広い signal なのか、candidate family / well / distance の base error と risk 飽和で
説明されるのかを、保存済み生成物だけで切り分ける。

## 制約

- Route: `pf_beam`。exp250 の PF/Beam candidate corridor 診断そのものを監査する。
- 入力は exp250 Stage 1 の candidate-segment、group、by-well、summary に固定する。
- exp250 Stage 1、PF/Beam、corridor、candidate、model、control を再実行しない。
- threshold / slack / segment grid、near 専用 rule、candidate prune / replacement、
  ML feature 化、raw-test inference、submission は行わない。
- real / shuffled の比較は同一 key と同一 bad/good weight を fail-closed で確認する。
- gzip 入力は decompressed content SHA を主証拠にする。
- Kaggle CPU / internet off / single process で初回 full readout を実行する。

## 受け入れ基準

- distance ごとの pooled paired AUC と、candidate family 内だけの conditional AUC を保存する。
- 0--100 ft の評価 weight、bad rate、全 distance に対する weight share を保存する。
- candidate family x well の real/shuffled AUC 差と、pair-mass による加法的寄与を保存する。
- variant / family ごとの risk=1.0 sample・評価 weight 飽和率を保存する。
- 入力 path / bytes / SHA / schema、出力 SHA、行数、well 数を summary に残す。
- train notebook は self-contained な読めるセル構成とし、inference / submission を fail-closed にする。
- 結果は診断としてのみ記録し、exp250 の不採用判断や route anchor を変更しない。

