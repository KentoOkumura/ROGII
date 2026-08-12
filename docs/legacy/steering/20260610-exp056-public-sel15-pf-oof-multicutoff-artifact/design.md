# 設計

## アプローチ

`exp029` の generator をコピーして、run config を all wells / cutoffs `[0.45, 0.65, 0.82]` / 16 seeds / 250 particles / gzip output に差し替える。コードは既に複数 cutoff の CLI / config に対応しているため、主な実装は実験メタデータ、notebook、記録、Kaggle kernel id、summary の multicutoff 化にする。

実行前 validation では local smoke を `--debug-n-wells 1 --cutoffs 0.45,0.65,0.82 --n-seeds 2 --n-particles 20` で行い、3 cutoff の rows と schema を確認する。full artifact は Kaggle train notebook で生成する。

## 実験範囲

- 対象実験: `exp056_public_sel15_pf_oof_multicutoff_artifact`
- Route: `pf_beam`
- 親実験: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: cutoff fractions を `[0.45, 0.65, 0.82]` に拡張、run label、kernel id/title、生成物説明
- 固定する変数: public sel15 PF/Beam primitive、selector scales、16 seeds、250 particles、output schema、leakage policy

## リスク

- リークリスク: cutoff 以降の `TVT_input` や train-only formation/geology columns を PF/Beam 入力に戻すと成立しない。`make_pseudo_hidden()` の mask と出力 target 専用扱いを維持する。
- CV/LB 不一致リスク: この実験は生成物作成だけであり、CV/LB 改善を主張しない。下流では 0.65 control rows を残す比較と総 row budget を揃える比較を分ける。
- ランタイム/メモリリスク: cutoff が 3 倍になるため、rows / gzip size / Kaggle runtime が exp029 v3 の概ね 3 倍近くになる。smoke と full Kaggle run を分け、output compression を gzip のまま維持する。
