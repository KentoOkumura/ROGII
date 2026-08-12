# 設計

## アプローチ

paired typewell の `TVT`/`GR` 曲線を horizontal evaluation zone の `GR` と照合する。known prefix から得た recent TVT slope を prior として、typewell TVT 上の bounded shift を探索し、normalized cross-correlation の score、confidence、ambiguity、matched typewell GR、anchor からの TVT delta を特徴化する。

## 実験範囲

- 対象実験: `exp008_gr_ncc_matcher`
- 親実験: `exp007_hard_well_router`
- 変更する変数: GR NCC feature set の有無、追加先 feature set (`no_gr_signal` / `all`)
- 固定する変数: GroupKFold、seed、HGB model params、sampling caps、last-anchor residual target、residual shrink

## Variants

| Variant | 内容 |
| --- | --- |
| `control_exp002_all` | exp002 all-GR residual model の再実行 |
| `control_exp003_no_gr` | exp003 selected no-GR feature set の再実行 |
| `gr_ncc_no_gr_multi` | no-GR feature set に multi-scale NCC features を追加 |
| `gr_ncc_all_multi` | all-GR feature set に multi-scale NCC features を追加 |

## リスク

- リークリスク: true `TVT` を alignment に使うと fold leakage になるため、known `TVT_input` prefix と typewell curve だけで shift search する。
- CV/LB 不一致リスク: public test の 3 visible wells と hidden test の well distribution が異なる可能性があるため、control rows と well-level metrics を必ず残す。
- ランタイム/メモリリスク: NCC search は feature construction を重くするため、candidate は `w25_r150` と `w75_r300` の 2 scale に限定する。
- 欠損リスク: GR 欠損や typewell 欠損時は NCC features を neutral default に落とし、既存 feature set を壊さない。
