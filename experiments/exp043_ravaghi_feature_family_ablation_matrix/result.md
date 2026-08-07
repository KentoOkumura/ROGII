# exp043_ravaghi_feature_family_ablation_matrix 結果

## 状態

Kaggle train notebook version 1 で完了。

## 結果

- 行数: 1,782,279
- wells: 773
- 最良の報告用比較基準: `pf090_hold010`, RMSE 15.089532
- 最良特徴候補: `base_plus_ncc_gr_match_pf_context_bucket_shrink`
  - original-fold RMSE: 15.634436
  - well-hash RMSE: 15.485651
- 同候補の raw 版:
  - original-fold RMSE: 15.526227
  - well-hash RMSE: 15.406654
- base 比較基準: `base_geometry_bucket_shrink`
  - original-fold 差分: -3.454973
  - well-hash 差分: -3.473667
- same-surface ML control: `exp026_regenerated_bucket_shrink`
  - original-fold RMSE: 16.483627
  - well-hash RMSE: 16.429613
  - 最良特徴候補との差分: -0.849191 / -0.943962
- PF 直接比較基準:
  - `public_pf_selector`: 15.172636
  - `pf090_hold010`: 15.089532
- LB: 未提出

## 解釈

同一 split での横並び比較により、Ravaghi 由来特徴は弱い base geometry 比較基準に対する単体 LightGBM では有効だと確認できた。最も強かった特徴条件は NCC/GR match と PF 不確実性 context の組み合わせで、exact beam 単独や spatial proxy 単独ではなかった。

ML route の同一 surface 比較では、最良特徴候補は `exp026_regenerated_bucket_shrink` を original-fold で -0.849191、well-hash で -0.943962 上回った。一方、すべての単体 LGBM 特徴候補は public PF の直接比較基準より弱い。したがって、これは直接置き換え候補ではなく、推論側へ移植しない。

この評価条件では raw variants が bucket-shrink variants より強いことが多く、特に `base_plus_ncc_gr_match_pf_context_raw` が該当する。固定 exp014 bucket shrink は、これらの Ravaghi 特徴モデルには合っていない可能性がある。

## 次

`exp043` は提出せず、推論側にも移植しない。この特徴量群比較は、信頼度 / 重み調整特徴、または XGBoost/CatBoost など別モデル種類の検証材料としてだけ使う。見えない test 風データ上の直接比較基準は `pf090_hold010` のまま維持する。
