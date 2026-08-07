# exp061_seedbag_anchor_model_diff_distance_gate

## 概要

`exp054` seed-bag pseudo-tail を anchor にし、`exp059` raw model-diff 予測との差分だけを距離 bucket 別に小さく混ぜる実験。

- route: `ml_model`
- parent: `exp059_pf_model_diff_foldsafe_surface_shrink`
- seed-bag anchor: `exp054_pseudo_tail_seed_bagging_inference_submit`
- status: submitted_complete

## 状態

Kaggle train v1, inference v1, and code submission completed.

## 仮説

exp059 raw model-diff は PF/Beam-vs-model の有効な補正信号を持つが、全面置換では exp054 seed-bag anchor より Public LB が悪い。exp054 を anchor として維持し、near/mid distance bucket だけに小さい alpha で model-diff 補正を入れれば、far bucket の悪化を抑えつつ改善分だけを拾える可能性がある。

## 候補

```text
seedbag_gate_pred = exp054_anchor + alpha(distance_bucket) * (raw_model_diff_pred - exp054_anchor)
```

Profiles:

- `near_mid_a0p25_far0`
- `near_mid_a0p50_far0`
- `global_a0p25`

`rows_2500_plus` は exp059 の弱点 bucket なので、near/mid profile では alpha 0 にして exp054 anchor へ戻す。

## 検証方針

exp029 pseudo-test surface 上で exp059 と同じ fold-safe source generation を使う。original-fold と well-hash holdout の RMSE、distance bucket RMSE、split RMSE を比較し、`exp054_foldout_control` と `lgbm_capacity_pf_model_diff_foldsafe_raw` の両方に対する差分を見る。

## 所見

Kaggle train v1 completed. Selected candidate is
`lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
with 14.872556 original-fold RMSE and 14.737595 well-hash RMSE.
The best pure model-diff gate was
`lgbm_capacity_pf_model_diff_foldsafe_seedbag_gate_near_mid_a0p50_far0`
with 14.838812 original-fold RMSE and 14.791874 well-hash RMSE.

Public LB is 11.826 (`ref=53581056`). This improves exp054 11.856 by -0.030,
but does not beat exp039 11.740.

## ファイル

- train notebook: `exp061_seedbag_anchor_model_diff_distance_gate_train.ipynb`
- inference notebook: `exp061_seedbag_anchor_model_diff_distance_gate_inference.ipynb`
- train helper: `pf_model_diff_model_audit.py`
- inference helper: `pf_model_diff_inference.py`
