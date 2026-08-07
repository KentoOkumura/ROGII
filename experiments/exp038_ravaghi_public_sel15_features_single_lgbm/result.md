# exp038_ravaghi_public_sel15_features_single_lgbm 結果

## 状態

Kaggle train version 1 completed.

## 実行

- Kernel: `kentookumura/exp038-ravaghi-sel15-lgbm-train`
- URL: `https://www.kaggle.com/code/kentookumura/exp038-ravaghi-sel15-lgbm-train`
- Output: `/tmp/kaggle-output/exp038_ravaghi_public_sel15_features_single_lgbm/train_v1`
- Rows: 1,782,279
- Wells: 773

## 結果

| Audit | Best candidate | RMSE |
| --- | --- | ---: |
| original-fold | `pf090_hold010` | 15.089532 |
| well-hash | `pf090_hold010` | 15.089532 |

| Candidate | original-fold RMSE | well-hash RMSE | 解釈 |
| --- | ---: | ---: | --- |
| `base_geometry_bucket_shrink` | 19.089409 | 18.959318 | single-LGBM base reference |
| `base_plus_pf_prediction_bucket_shrink` | 15.850147 | 15.820850 | selected single-LGBM feature candidate |
| `base_plus_pf_beam_diagnostics_bucket_shrink` | 15.998458 | 15.604067 | well-hash は良いが original-fold で PF-only に劣る |
| `exp026_regenerated_bucket_shrink` | 16.483627 | 16.429613 | exp026-style regenerated 基準 on exp029 surface |
| `public_pf_selector` | 15.172636 | 15.172636 | feature model より強い public PF control |
| `pf090_hold010` | 15.089532 | 15.089532 | overall best control |

## 解釈

Ravaghi/public sel15 PF prediction features は、base single LightGBM reference に対して original-fold で -3.239262、well-hash で -3.138468 改善した。一方、`public_pf_selector` と `pf090_hold010` には届かなかった。

このため、PF/Beam 候補値を単体 LightGBM に入れる価値はあるが、現時点では public PF candidate を置き換えるほど強くない。次は LightGBM 単体の inference port ではなく、PF confidence / divergence を直接 candidate selection や postprocess gate に使う方向を優先する。
