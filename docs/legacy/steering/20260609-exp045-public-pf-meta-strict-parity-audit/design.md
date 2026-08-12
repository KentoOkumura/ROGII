# 設計

## アプローチ

`exp035_public_sel15_pf_meta_inference_port` を最小コピーし、見えない test well 用処理 の PF diagnostics だけを exp029 parity 設定へ下げる。exp026-style anchor fit、exp029 sampled rows への anchor 再生成、Ridge meta residual fit、visible branch preserve は exp035 と同じにする。Kaggle output の row-level diff と summary に PF parity 設定、prediction range、meta residual 分布を残す。

## 実験範囲

- 対象実験: `exp045_public_pf_meta_strict_parity_audit`
- Route: `pf_beam`
- 親実験: `exp035_public_sel15_pf_meta_inference_port`
- 変更する変数: 見えない test well 用 PF 診断値 の `n_particles=250`、`n_seeds=16`、実験名、kernel id、監査 summary。
- 固定する変数: exp026-style anchor、exp034 selected Ridge meta candidate、meta feature列、selector rules、beam configs、visible physical branch、postprocess。

## リスク

- リークリスク: 見えない test の評価区間 TVT は使わず、known `TVT_input` prefix、typewell、MD/X/Y/Z/GR、PF/Beam diagnostics だけを使う。meta training target は exp029 train well の途中以降を隠した疑似 test 生成物 内に限定する。
- CV/LB 不一致リスク: CV は新規に計算しない。exp034 の train well の途中以降を隠した疑似 test evidence と exp045 Public LB / 見えない test well 用処理の summary の差分を診断材料に限定する。
- ランタイム/メモリリスク: PF seeds/particles を exp035 の 128/500 から 16/250 に下げるため、inference runtime は短くなる見込み。ただし meta training rows は exp035 と同じ sampling cap のまま。
