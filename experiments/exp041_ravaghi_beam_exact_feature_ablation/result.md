# exp041_ravaghi_beam_exact_feature_ablation 結果

## 状態

Kaggle train notebook version 1 で完了。

## 検証

- 入力: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- Kaggle kernel: `kentookumura/exp041-ravaghi-beam-exact-train` version 1
- 監査:
  - leave-one-original-fold-out
  - well-hash holdout
- 比較基準: `base_geometry_bucket_shrink`
- 報告用比較対象: `last_anchor`, `public_pf_selector`, `pf090_hold010`, `beam`
- 行数: 1,782,279
- wells: 773

## 結果

| candidate | original-fold RMSE | well-hash RMSE | base original との差分 | base well-hash との差分 |
| --- | ---: | ---: | ---: | ---: |
| `pf090_hold010` | 15.089532 | 15.089532 | -3.999877 | -3.869786 |
| `public_pf_selector` | 15.172636 | 15.172636 | -3.916773 | -3.786682 |
| `base_plus_beam_exact_disagreement_raw` | 15.391724 | 15.590140 | -3.697685 | -3.369178 |
| `base_plus_beam_exact_disagreement_bucket_shrink` | 15.527268 | 15.727948 | -3.562141 | -3.231370 |
| `base_plus_beam_exact_pf_context_raw` | 15.886148 | 15.636310 | -3.203261 | -3.323008 |
| `base_plus_public_beam_aggregate_raw` | 15.973794 | 15.987131 | -3.115615 | -2.972187 |
| `base_plus_public_and_exact_beam_raw` | 16.018399 | 15.785438 | -3.071010 | -3.173881 |
| `base_plus_public_beam_aggregate_bucket_shrink` | 16.123567 | 16.132100 | -2.965842 | -2.827218 |
| `exp026_regenerated_bucket_shrink` | 16.483627 | 16.429613 | -2.605782 | -2.529705 |
| `base_geometry_bucket_shrink` | 19.089409 | 18.959318 | 0.000000 | 0.000000 |

設定上の bucket-shrink 系から選んだ候補:

- `base_plus_beam_exact_disagreement_bucket_shrink`
- original-fold RMSE: 15.527268
- well-hash RMSE: 15.727948
- `base_geometry_bucket_shrink` から -3.562141 / -3.231370 改善
- `exp026_regenerated_bucket_shrink` から -0.956360 / -0.701665 改善
- `public_pf_selector` には +0.354631 / +0.555312 届かない
- 固定 `pf090_hold010` には +0.437736 / +0.638416 届かない

## 解釈

Ravaghi exact beam の食い違い特徴は、exp029 の見えない test 風データでは追加特徴として効いている。単体 LGBM の弱い比較基準を大きく改善し、再生成した exp026-style 比較基準も両方の holdout 監査で上回った。

ただし、public PF の直接比較基準よりは弱い。`pf090_hold010` と `public_pf_selector` は、どちらの監査でも選択した特徴モデルより良い。また `base_plus_beam_exact_disagreement_raw` は bucket-shrink 版より強いため、次に使うなら exact beam の食い違いを信頼度 / 食い違い特徴、または raw 候補入力として扱い、即時の推論移植には進めない。
