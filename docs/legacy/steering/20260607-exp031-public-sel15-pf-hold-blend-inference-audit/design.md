# 設計

## アプローチ

`exp027_public_replay_needless090_sel15_spread3` を source にして公開 notebook の実装を保つ。末尾の submission 生成部で、見えない test well の `tvt_selector` を `0.90 * tvt_selector + 0.10 * last_known_tvt` に置き換える。`tvt_selector` は `exp029` feature artifact の `pf_pred` に対応する公開 selector 予測として扱う。

監査用に、元 selector submission、row-level diff、summary JSON を同時に出力する。最終提出候補は `submission.csv`。

## 実験範囲

- 対象実験: `exp031_public_sel15_pf_hold_blend_inference_audit`
- Route: `pf_beam`
- 親実験: `exp027_public_replay_needless090_sel15_spread3`
- 変更する変数: 見えない test well の selector 出力に対する hold blend weight 0.10
- 固定する変数: PF particles/seeds/scales、Beam configs、selector bins、visible physical model、Kaggle metadata

## リスク

- リークリスク: `last_known_TVT_input` だけを使うため情報制約は hidden test と一致する。評価区間の真値や未来 TVT は使わない。
- CV/LB 不一致リスク: `exp030` は train well の途中以降を隠す cutoff 0.65 の OOF-like 診断であり、Public LB 8.781 の公開 replay に同じ改善が乗る保証はない。
- ランタイム/メモリリスク: 元 `exp027` と同じ PF/Beam 実行に row-level audit CSV が追加されるだけなので小さい。
