# 設計

## アプローチ

exp311群priorを初期分布とし、`intercept, log_scale`がrandom walkする2-state linear Gaussian modelを構成する。outer-trainだけでprocess noiseをempirical Bayes固定し、held-out wellではvisible prefixをKalman filterする。last640 maskでcausal extrapolationを評価し、smootherはprefix内診断に限定、boundary先には使わない。

## 実験範囲

- 対象: `exp318_typewell_group_time_varying_calibration_ssm`
- Route: `pf_beam`
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 変更: static group priorから2-state causal calibrationへ。
- 固定: state、transition family、mask horizon、fit境界、runtime/gain guard。
- 計算量: Stage 0/1各1 variant、model/booster/TVT decoder 0。

## 再現性設計

- 線形代数、well順、fold、初期値を固定しglobal RNGを使わない。
- group prior、process noise、filter state、boundary prediction、runtime profileのSHAを保存する。
- fixed 16-well microbenchmarkから全fold runtimeを外挿し、8.5h超なら科学実行前に停止する。

## リスクと停止条件

- visible prefixの校正driftがsuffixへ持続しない可能性が高い。
- Stage 0で可同定性またはruntimeがFAILならStage 1を作らない。
- exp295の位置×rate巨大stateやsoft-label structured trainingは再導入しない。
