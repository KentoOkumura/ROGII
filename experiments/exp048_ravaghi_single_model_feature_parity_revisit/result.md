# exp048_ravaghi_single_model_feature_parity_revisit 結果

## 状態

Kaggle train notebook version 1 で完了。

## 結果

- 行数: 1,782,279
- wells: 773
- original-fold 最良: `pf090_hold010`, RMSE 15.089532
- well-hash 最良: `base_plus_ncc_gr_match_pf_context_pf_blend_w0p30`, RMSE 15.019511
- direct PF controls:
  - `public_pf_selector`: 15.172636
  - `pf090_hold010`: 15.089532
- same-surface ML control:
  - `exp026_regenerated_bucket_shrink`: original-fold 16.483627 / well-hash 16.429613
- standard ML route anchors for reference:
  - usual CV anchor: `exp025_pseudo_tail_postprocess_cv_audit` fixed bucket shrink 12.870780
  - Public LB anchor: `exp039_ravaghi_single_lgbm_inference_submit` 11.740
- best single-LGBM/PF-blend candidate:
  - `base_plus_ncc_gr_match_pf_context_pf_blend_w0p30`
  - original-fold RMSE: 15.122880
  - well-hash RMSE: 15.019511
- strict supported candidates: none
- LB: 未提出

## 解釈

固定 PF blend により、`base_plus_ncc_gr_match_pf_context_pf_blend_w0p30` は well-hash で
`pf090_hold010` を 0.070021 上回ったが、original-fold では `pf090_hold010` より 0.033348 悪い。
したがって、設定した成功条件である `public_pf_selector` と `pf090_hold010` の両方を
original-fold / well-hash の両方で上回る候補はない。

ML route 内の比較としては、同一 `exp029` 見えない test 風 surface 上の
`exp026_regenerated_bucket_shrink` からは original-fold で -1.360747、well-hash で -1.410102
改善している。ただしこの surface は通常 CV 12.870780 や Public LB 11.740 とは評価条件が違う。
過去の exp031/033/035/047 でも、この surrogate surface の改善は Public LB に転移しない例があるため、
この改善だけで ML route anchor 更新や inference port はしない。

Ravaghi 由来特徴は、単体 LightGBM の弱い base geometry には効くが、LightGBM 直接置き換えや
固定 blend としては direct PF controls を安定して超えない。推論 port / submit は行わない。

## 次

Ravaghi feature は LightGBM 直接置き換えでは停止する。後続で使う場合は、XGBoost/CatBoost の
入力候補、または PF/Beam confidence-only feature として限定的に扱う。
