# 設計

## アプローチ

`exp046` の train-side surrogate audit を土台にする。`exp029` の疑似 test PF/Beam rows を読み、original-fold、well-hash、stratified-group fold の各 audit split で `exp026` pseudo-tail bucket-shrink anchor を fold-safe に再生成する。その上で、候補生成だけを gate-only に差し替える。

候補は固定 gate と learned gate に分ける。固定 gate は `exp026` から public PF / `pf090_hold010` / beam へ 0.10-0.30 だけ寄せる。learned gate は `optimal_weight` または `candidate_wins` を target にし、Ridge / small HistGradientBoosting で `w` を予測する。予測された `w` は必ず `[0, max_weight]` に clip し、最終予測は `base + w * (candidate - base)` に限定する。

## 実験範囲

- 対象実験: `exp047_public_pf_beam_gate_only_audit`
- Route: `pf_beam`
- 親実験: `exp046_hidden_branch_surrogate_audit`
- 変更する変数: PF/Beam の使い方を自由な残差補正から clipped gate-only へ変更する。
- 固定する変数: `exp029` PF/Beam pseudo-test 入力、`exp026` anchor 手順、audit split、distance bucket、exp044 stratified metadata。

## 監査出力

- overall RMSE と reference delta
- distance bucket / split / metadata bucket の segment RMSE
- well-level RMSE
- candidate vs reference diff metrics
- gate weight stats
- exp026 anchor 再生成 source summary

## リスク

- リークリスク: `target_tvt` は gate target と scoring に使うが、feature には入れない。error diagnostic columns は feature に入れない。
- CV/LB 不一致リスク: exp046 surrogate で良く見えた候補が実 Public LB で失敗済みなので、gate-only audit の改善だけで submit へ進まない。
- ランタイム/メモリリスク: full 実行は split system ごとに exp026 anchor を再生成するため重い。まず `--max-wells` smoke で schema と実行経路を確認し、正式実行は Kaggle train notebook を正とする。
