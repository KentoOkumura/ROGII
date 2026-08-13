# exp019_pf_beam_candidate_quality_audit 結果

## 結論

Kaggle version 4 が完了。PF/beam direct candidates と PF/beam feature model は raw LightGBM 基準を支持できず、PF/beam 再投入はしない。

## 主要指標

| Metric | Value |
| --- | --- |
| Raw LightGBM no-GR CV | 13.549257 |
| Best direct candidate | `raw_lightgbm_no_gr` |
| Best direct candidate CV | 13.549257 |
| Best direct PF delta vs raw | 0.000000 |
| exp015 PF feature model mean well delta vs control | +0.648761 |
| Rows / wells | 3,783,989 / 773 |
| Skipped wells | 0 |

## 解釈

- Full-row best は raw LightGBM のまま。PF-derived full-row candidates は `pf_hold_mean_blend` でも 19.142388 で raw 13.549257 より +5.593130 悪い。
- `pf_best` は 114.654448 で raw より +101.105190 悪く、scale selector の best path は direct prediction として壊れている。
- PF confidence / best scale / GR missing / long eval / high Z span / steep trajectory の各 group でも raw が最良で、PF を狭い group に router 投入する根拠はない。
- `exp015` PF feature model は 310/773 wells では改善したが、mean well delta は +0.648761 と悪化側。top-help を追うよりも、PF 由来 signal は破棄する判断が妥当。
- rows 0-49 では `recent_linear` 0.796588、`last_anchor` 0.960110 が raw 3.231596 より良く、rows 50-249 でも `recent_linear` 3.615510 が raw 3.829747 より良い。ただしこれは PF ではなく near-row 基準/recent behavior なので、既存の postprocess / distance-aware training audit の範囲で扱う。

## 次のアクション

`backlog/KAGGLE_DIRECTION.md` から PF/beam candidate quality audit を削除し、次は距離 bucket の raw residual bias/variance と near-row damping/training weight を診断する。
