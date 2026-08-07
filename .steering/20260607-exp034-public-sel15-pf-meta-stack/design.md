# 設計

## アプローチ

`exp029` の PF/Beam feature rows に対して、`exp026` と同じ pseudo-tail LightGBM + fixed bucket shrink を audit split ごとに再生成する。これらの rows は train well の途中以降を隠し、本番 test 風に予測させたもの。その予測を `exp026_pred` として追加し、PF/Beam diagnostics と組み合わせた 2nd stage を比較する。

2nd stage は保守的な候補に限定する。

- fixed blend: `exp026_pred` と `pf_pred` / `beam_pred` の小さな blend。
- ridge residual meta: `target_tvt - exp026_pred` を clipped/shrunk ridge で学習する。
- shallow HGB residual meta: 非線形候補の上限確認用。採用は両 holdout で安定した場合だけ。

## 実験範囲

- 対象実験: `exp034_public_sel15_pf_meta_stack`
- Route: `pf_beam`
- 親実験: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: exp026-style anchor と PF/Beam features の 2nd stage。
- 固定する変数: exp026 pseudo-tail recipe、exp014 bucket shrink params、exp029 PF/Beam OOF-like artifact、well-level validation policy。

## リスク

- リークリスク: base prediction を validation well を含む model で作ると stacking leakage になる。audit split ごとに validation wells を除外して `exp026_pred` を作る。
- CV/LB 不一致リスク: exp031/exp033 で train well の途中以降を隠した疑似 test 改善が 見えない test well 評価の LB に転移しなかったため、supported candidate が出ても別 inference-port audit なしで提出しない。
- ランタイム/メモリリスク: exp026-style model を original-fold と well-hash で再生成するため Kaggle train は重い。ローカルは `--max-wells` smoke に限定し、full は Kaggle Notebook で実行する。
