# exp015_public_pf_beam_scale_selector_features 結果

## 結論

PF/beam scale selector features の add-only 追加は採用しない。Kaggle full CV で `pf_beam_no_gr` は 14.442743 となり、raw control `control_lightgbm_no_gr` 13.549257 から 0.893486 悪化した。

## Kaggle Train

- Kernel: `kentookumura/exp015-pf-beam-train`
- Version: 1
- Status: COMPLETE
- Runtime: log timestamp 約 1,180 秒
- Output: `/tmp/kaggle-output/exp015_public_pf_beam_scale_selector_features/train`

## スコア

| Variant | Feature set | CV | mean fold RMSE | delta vs control |
| --- | --- | ---: | ---: | ---: |
| `control_lightgbm_no_gr` | `no_gr_signal` | 13.549257 | 13.521370 | 0.000000 |
| `pf_beam_no_gr` | `no_gr_signal_plus_pf_beam` | 14.442743 | 14.401690 | +0.893486 |

Postprocess artifact:

| Candidate | Source | RMSE | Note |
| --- | --- | ---: | --- |
| `raw_lightgbm_no_gr` | `pf_beam_no_gr` | 14.442743 | selected variant raw |
| `distance_bucket_shrink_fit` | `pf_beam_no_gr` | 14.394712 | same-OOF fit; still worse than raw control |

## Group Notes

`model_group_summary.csv` でも PF/beam features は主要 group で一貫して悪化した。

| Group | control mean RMSE | PF/beam mean RMSE | delta |
| --- | ---: | ---: | ---: |
| all | 10.901288 | 11.550049 | +0.648761 |
| high_gr_missing | 10.491004 | 11.145623 | +0.654619 |
| long_eval | 11.942640 | 12.787737 | +0.845097 |
| steep_trajectory | 12.380045 | 13.094070 | +0.714025 |

## 解釈

今回の deterministic PF/beam snapshot は、LightGBM が使いやすい補助 signal ではなく、既存 no-GR residual model にノイズを足した。公開 notebook の強い PF/beam route を add-only feature として浅く入れるだけでは不十分で、候補 path 自体の品質または routing/feature pruning が必要。

ただし、control が 13.549257 を完全再現したため、実験比較面は壊れていない。

## 次

PF/beam add-only は backlog から外す。次は `public_postprocess_ablation` を優先し、fade-in / hold blend / alpha-tau / SG smoothing を raw 基準 と exp014 held-out postprocess audit の評価面で切り分ける。
