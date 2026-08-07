# exp211_affine_calibrated_gr_observation_pfbeam

## 目的

`affine_calibrated_gr_observation_pfbeam` backlog を実装する。known prefix だけで `horizontal_GR ~= a * typewell_GR(TVT_input) + b` を robust fit し、affine calibrated GR を PF/Beam の observation likelihood に入れた場合に、raw GR baseline より候補品質が改善するかを train-side pseudo-tail で確認する。

## 状態

- `completed_train_side_diagnostic_no_submit`
- Kaggle train v1 完了。
- inference / submit は対象外。

## 仮説

prefix affine calibration で well ごとのGR scale/offsetずれを補正できれば、raw GR observation よりPF/Beamのdepth matchingが安定する。特に public notebook 群が使う raw GR residual-scale control では吸収しきれない offset/scale差に対して、affine observation が候補品質やPF diagnosticsを改善する可能性がある。

## 範囲

- Route: `pf_beam`
- 親: `exp189_denoised_gr_pfbeam_generation_audit`
- 参照: exp072 full replay feature cache、exp170 heel calibration audit、public notebook catch-up memo
- 検証: exp072 の `TVT_input_missing_equivalent_exp063_rows`
- 2x2 variants:
  - `raw`: raw GR + classic transition
  - `affine`: affine calibrated GR + classic transition
  - `raw_structural`: raw GR + weak prefix structural prior
  - `affine_structural`: affine calibrated GR + weak prefix structural prior
- 対象外: LightGBM 学習、direct replacement、inference port、submit

## 検証方針

- target wells は true error を使わず、eval row / md_since coverage で最大 64 wells を選ぶ。
- raw / affine variants は同じ PF seed、particles、transition noise、beam width で比較する。
- affine fit は known prefix のみで行い、fallback rate、slope、intercept、prefix RMSE、fallback reason を記録する。
- primary baseline は `pf_raw_lik_mean`。加えて exp072 reference candidates (`pf_ancc`, `pf_z`, `beam_mean`, `likpf_mean`) と比較する。

## 所見

affine calibration は 64/64 wells で fallback なしにfitできたが、PF/likelihood-PF の direct observation としては悪化した。`pf_affine_lik_mean` は primary `pf_raw_lik_mean` 18.640063 から 21.184758 へ +2.544695 悪化し、`pf_affine_structural_lik_mean` も 21.143708 で悪化した。

Beam では `beam_affine_top1` が raw Beam 18.339188 から 18.065010 へ -0.274177 改善したが、best non-oracle は既存 `exp072_pf_ancc` 17.494197 のままで、max well regression も +20.781499 残る。direct replacement / inference port / submit は行わない。

使う場合は、affine slope/intercept、prefix RMSE、raw-vs-affine disagreement、oracle headroom を selector / confidence feature の診断材料に限定する。

## 生成物

Kaggle train 実行後、`artifacts/` に candidate metrics、variant delta metrics、bucket/group/by-well metrics、PF diagnostics、target wells、row candidates、summary JSON を保存する。
