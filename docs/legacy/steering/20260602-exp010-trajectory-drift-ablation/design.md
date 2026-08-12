# 設計

## アプローチ

last-known TVT anchor の residual model に、hidden test でも利用できる trajectory drift feature group を追加する。`MD`, `X`, `Y`, `Z` から direction、slope、interaction を作り、exp003 の no-GR control と exp002 の all-GR control に対して ablation する。

## 実験範囲

- 対象実験: `exp010_trajectory_drift_ablation`
- 親実験: `exp009_formation_surface_guide`
- 変更する変数: model feature set
- 固定する変数: split、metric、HGB params、sampling caps、residual shrink、GR NCC disabled、formation guide disabled

## Feature Groups

- direction: anchor / prefix / step azimuth sin/cos、prefix からの turn、final-axis signed projection、`delta_x/y/z` sign。
- slope: `dX/dMD`、`dY/dMD`、`dXY/dMD`、`dZ/dXY`、inclination、known prefix trajectory slope。
- interaction: recent TVT slope と `delta_md` / `anchor_dz_dmd`、prefix-vs-eval slope delta、eval progress との interaction。

## リスク

- リークリスク: trajectory は hidden test で利用可能な列だけを使う。target-derived feature は known prefix の `TVT_input` に限定する。
- CV/LB 不一致リスク: public-like wells では GR feature が効く可能性があるため、no-GR full だけでなく all-GR full も比較する。
- ランタイム/メモリリスク: feature 数増加のみで model / sampling caps は固定するため、exp009 と同程度に収まる想定。
