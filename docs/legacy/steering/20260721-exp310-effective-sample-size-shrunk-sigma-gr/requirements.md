# 要件

## 依頼

- `effective_sample_size_shrunk_sigma_gr`を`exp310`として新規作成し、設計を確定する。
- exp307 finite-MAD `σ_GR`を、prefix GR残差の自己相関から求めた有効標本数に応じてcross-well priorへ縮約する。
- 今回はdesign-onlyとし、実装、Kaggle実行、inference、submissionは行わない。

## 仮説

GR残差は強く系列相関しているため、finite pair数をそのままscale推定の信頼度とみなすとwell別MADを過信する。有効標本数が少ないwellだけglobal priorへ縮約すれば、exp307の適応性を保ちながらtail regressionを抑えられる。

## 制約

- Routeは`pf_beam`、親は`exp307_finite_only_robust_sigma_gr`とする。
- exp307の全promotion gate PASSが必須。実行優先はexp307 scale uncertainty診断でtriggerが成立した場合に限る。
- contiguous finite residual runだけからlag 1--20の自己相関を集計し、`tau=1+2*sum(max(rho_k,0))`、`n_eff=clip(n/tau,1,n)`に固定する。
- priorは対象wellを除く772 wellsのfinite-MAD中央値、`alpha=n_eff/(n_eff+50)`、log-scale shrinkage、clip `[10,60]`とする。
- exp307の残差定義、MAD、fallback、evaluation GR、HMM decoderを固定する。
- lag上限、positive-sequence式、prior、k=50、linear/log shrink、clipのgridを行わない。
- exp308 downweight、exp309 transition noiseを混ぜない。
- 1 variant x 773 = 773 HMM well-runs、model/LightGBM/fold/booster/PF/Beamは0、parent再実行0とする。

## 受け入れ基準

- exp307 dependency/SHA、contiguous-run ACF、n_eff/prior/shrink formula、truth late-joinをPASSする。
- target-free triggerとして、`n_eff/n`中央値0.5以下、または`n_eff<50`のwell比率20%以上のいずれかを満たす。満たさなければ実行せずcloseする。
- 実行時はexp307 primaryよりoverall RMSEを0.03 ft以上、4/5 foldsで改善する。
- 1000+、hidden-like 2面、by-well p95を悪化させず、worst regressionを+0.25 ft以下にする。
- fixed LikPF 50:50をparent blendから悪化させない。
- FAIL後のlag/k/prior/clip/grid、Student-t/mixture、exp308/309とのcombined rescueへ進まない。

## 次のアクション

exp307の全promotion gate PASSとtarget-free trigger、別途実装承認が揃うまでブロックする。
