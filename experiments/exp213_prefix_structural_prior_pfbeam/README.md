# exp213_prefix_structural_prior_pfbeam

## 目的

`prefix_structural_prior_pfbeam` backlog を実装する。known prefix の `TVT_input + Z` を structural surface state として MD に対して robust fit し、PF/Beam の初期状態と遷移 prior にだけ使う。観測 likelihood は raw GR のまま固定する。

## 状態

- `completed_train_side_diagnostic_no_submit`
- Kaggle train v1 完了。
- inference / submit は対象外。

## 仮説

pointwise GR matching だけでは datum alias や二峰性を解けないため、known prefix から見える structural surface の局所 slope / step delta を soft prior として入れると、PF/Beam がより自然な path を残せる可能性がある。exp142/200 の反省から hard window や強制 transition にはせず、mode diversity と top-K path を保存して診断する。

## 範囲

- Route: `pf_beam`
- 親: `prefix_structural_prior_pfbeam backlog`
- 実装親: `exp211_affine_calibrated_gr_observation_pfbeam`
- 参照: `exp072` full replay feature cache、`exp146`、`exp200`、`exp142`
- 検証: exp072 の `TVT_input_missing_equivalent_exp063_rows`
- variants:
  - `raw`: raw GR + classic transition
  - `structural_slope_only`: raw GR + prefix structural slope / step-delta prior
  - `structural_weak`: raw GR + weak absolute structural prior
  - `structural_base`: raw GR + base absolute structural prior
- 対象外: LightGBM 学習、direct replacement、inference port、submit

## 検証方針

- target wells は true error を使わず、eval row / md_since coverage で最大 64 wells を選ぶ。
- raw / structural variants は同じ PF seed、particles、transition noise、beam width で比較する。
- structural fit は known prefix のみで行い、surface slope、residual sigma、expected delta、velocity blend、Beam cost gap、path spread を記録する。
- primary baseline は `pf_raw_lik_mean`。加えて exp072 reference candidates (`pf_ancc`, `pf_z`, `beam_mean`, `likpf_mean`) と比較する。

## 所見

PF では structural prior が大きく悪化した。`pf_raw_lik_mean` RMSE 21.081279 に対し、`pf_structural_weak_lik_mean` は 28.230909、`pf_structural_base_lik_mean` は 29.564037、`pf_structural_slope_only_lik_mean` は 30.621856 だった。prefix surface の slope / delta prior が longtail で wrong path へ強く寄せた可能性が高い。

Beam では `structural_base` がごく小さく改善した。`beam_raw_top1` RMSE 18.339188 に対し `beam_structural_base_top1` は 18.312677、差分は -0.026510。全 distance bucket で小幅改善したが、best non-oracle は既存 `exp072_pf_ancc` RMSE 17.494197 のままで、direct replacement の根拠には足りない。

したがって direct PF/Beam generation 変更としては不採用。inference port / submit は行わず、残す場合は Beam top-K gap / path spread / structural disagreement などの confidence feature 材料に限定する。

## 生成物

Kaggle train 実行後、`artifacts/` に candidate metrics、variant delta metrics、bucket/group/by-well metrics、PF diagnostics、target wells、row candidates、summary JSON を保存する。
