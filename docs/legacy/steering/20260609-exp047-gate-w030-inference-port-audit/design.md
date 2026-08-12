# 設計

## アプローチ

`exp045_public_pf_meta_strict_parity_audit` の inference notebook を実装親にする。exp045 には exp026-style anchor 生成、public sel15 replay、visible public sample branch 維持、hidden PF selector 生成、audit summary 保存の経路が揃っているため、その構造を残して meta residual 学習と適用だけを取り除く。

hidden / unseen test well では、exp026 anchor と public PF selector prediction を作ったあと、固定 `w=0.30` で `exp026_anchor + w * (pf_pred - exp026_anchor)` を計算する。visible train well は従来の physical branch を使い、sample output が変わらない場合でも hidden branch summary で理由を確認できるようにする。

## 実験範囲

- 対象実験: `exp047_gate_w030_inference_port_audit`
- Route: `pf_beam`
- 親実験: `exp047_public_pf_beam_gate_only_audit`
- 実装親: `exp045_public_pf_meta_strict_parity_audit`
- 変更する変数: hidden / unseen test well 用 branch の候補を Ridge meta residual から固定 `exp026_to_pf_gate_w0p30` に変える。
- 固定する変数: exp026-style anchor、public visible physical branch、PF selector settings、distance bucket shrink、Kaggle offline metadata。

## 監査出力

- `submission.csv`
- `public_sel15_exp026_anchor_submission.csv`
- `public_sel15_gate_w030_diff.csv`
- `public_sel15_gate_w030_summary.json`

## リスク

- リークリスク: hidden / unseen test well の評価区間 TVT は読まない。gate は固定重みなので test side で target を学習しない。
- CV/LB 不一致リスク: exp031/033/035/045 で surrogate 改善と Public LB が乖離済み。submit する場合は 1 回に限定し、悪化時は public PF gate inference port を停止する。
- ランタイム/メモリリスク: meta model fit を削除するため exp045 より軽いが、PF selector と exp026 anchor fit は残る。Kaggle inference で確認する。
