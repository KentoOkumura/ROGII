# 設計

## アプローチ

`exp033` の inference port を土台にし、residual correction branch を `exp034` の selected Ridge meta residual branch に置き換える。exp034 の `baseline.py` と `pseudo_tail_augmentation.py` を同梱し、Kaggle inference notebook 内で exp026-style pseudo-tail anchor を fit する。

inference notebook は次を行う。

- train wells から exp026-style pseudo-tail anchor model を fit。
- exp029 train well の途中以降を隠した疑似 test feature artifact から sampled rows を読み、各 `(well_id, cutoff_row)` に exp026 anchor prediction を再生成。
- `target_tvt - exp026_pred` を target として Ridge meta residual model を fit。
- hidden test wells に public sel15 PF/Beam diagnostics と exp026 anchor feature を作成。
- hidden test wells に `exp026_anchor + 0.75 * clip(meta_residual, -60, 60)` を適用。
- visible test wells は既存 physical-model prediction のままにする。

## 実験範囲

- 対象実験: `exp035_public_sel15_pf_meta_inference_port`
- Route: `pf_beam`
- 親実験: `exp034_public_sel15_pf_meta_stack`
- 変更する変数: 見えない test well 用処理の base/correction、meta model fit、audit artifact 名
- 固定する変数: public sel15 PF/Beam selector、visible physical branch、competition input、Kaggle offline 実行前提

## リスク

- リークリスク: exp029 artifact は train well の途中以降を隠した疑似 test rows だけを target 付きで使う。見えない test well の TVT は読まない。
- CV/LB 不一致リスク: exp031/exp033 の 見えない test well 評価の LB 悪化実績があるため、output audit と submit-check なしで提出しない。
- ランタイム/メモリリスク: inference notebook 内で exp026 anchor model fit と sampled-row anchor regeneration を行うため、exp033 より重い。`META_MAX_TRAIN_ROWS` と bucket cap で Ridge training rows を制限する。
