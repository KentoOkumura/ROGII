# exp040_ravaghi_pf_ancc_pfz_feature_ablation 結果

## 状態

Kaggle train notebook version 1 で完了。

## 検証

- 入力: `experiments/exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
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
| `base_plus_pf_ancc_pfz_core_raw` | 15.859121 | 15.607618 | -3.230288 | -3.351700 |
| `base_plus_pf_ancc_pfz_uncertainty_raw` | 15.865819 | 15.532799 | -3.223590 | -3.426519 |
| `base_plus_pf_uncertainty_raw` | 15.883552 | 15.492413 | -3.205857 | -3.466905 |
| `base_plus_pf_ancc_pfz_uncertainty_bucket_shrink` | 16.011790 | 15.645900 | -3.077619 | -3.313418 |
| `base_plus_pf_uncertainty_bucket_shrink` | 16.023513 | 15.596025 | -3.065896 | -3.363294 |
| `exp026_regenerated_bucket_shrink` | 16.483627 | 16.429613 | -2.605782 | -2.529705 |
| `base_plus_pf_z_proxy_bucket_shrink` | 17.050009 | 16.803471 | -2.039400 | -2.155847 |
| `base_geometry_bucket_shrink` | 19.089409 | 18.959318 | 0.000000 | 0.000000 |

設定上の bucket-shrink 系から選んだ候補:

- `base_plus_pf_ancc_pfz_uncertainty_bucket_shrink`
- original-fold RMSE: 16.011790
- well-hash RMSE: 15.645900
- `base_geometry_bucket_shrink` から -3.077619 / -3.313418 改善
- `exp026_regenerated_bucket_shrink` から -0.471838 / -0.783713 改善
- `public_pf_selector` には +0.839154 / +0.473264 届かない
- 固定 `pf090_hold010` には +0.922258 / +0.556368 届かない

## 解釈

Ravaghi PF ANCC/PFZ 風の代替特徴は、exp029 の見えない test 風データでは追加特徴として効いている。対応する bucket-shrink 特徴群は、どちらの holdout 監査でも単体 LGBM の弱い比較基準を上回り、選択候補は再生成した exp026-style 比較基準も上回った。

ただし、public PF の直接比較基準を置き換えるほど強くない。`pf090_hold010` と `public_pf_selector` は、どちらの監査でも選択した特徴モデルより良い。さらに、この評価条件では raw 特徴候補が bucket-shrink 候補より強い面があり、特に well-hash の `base_plus_pf_uncertainty_raw` が良い。次に使うなら、即時の推論移植や提出候補ではなく、信頼度 / 食い違い特徴、または重み調整の入力として扱う。
