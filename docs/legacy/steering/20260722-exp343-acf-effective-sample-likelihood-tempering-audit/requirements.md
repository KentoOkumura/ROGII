# 要件

## 依頼

- known-prefix GR residual ACFから有効標本数を推定し、系列相関で重複するemission evidenceをwell別にtemperする設計を確定する。
- 2026-07-23のユーザー依頼により、0-HMMのStage 0 ACF stability readoutだけを実装する。
- Kaggle package/push/run、Stage 1 HMM、inference、submissionは別承認まで行わない。
- 旧`exp320_typewell_group_noise_spectrum_whitening`の「相関したGR evidenceを重複加算しない」
  着眼点に対する独立後継として扱う。exp320はreopen / reparentしない。

## 制約

- Route: `pf_beam`。
- 科学的親/decoder controlは`exp281_exp226_residual_offset_exact_hmm_transition_probe`。negative referencesはexp232/305、設計参照はexp310とする。
- exp311/313/320のType Well群統計、group AR(1)、group-label transferには依存しない。
- ACFはraw finite known-prefix residualのcontiguous run内だけでlag `1--20`を計算し、missingをまたがない。
- finite residual 128未満または各lag pair 20未満はouter-train fold medianへfallbackする。
- `tau_raw=1+2*sum(max(rho_k,0))`、`alpha=n/(n+200)`、`tau_shrunk=exp(alpha*log(tau_raw)+(1-alpha)*log(tau_fold_median))`、`tau_eff=clip(tau_shrunk,1,4)`に固定する。
- Stage 0でfull prefixとlast-512 prefixの安定性を監査する。Stage 1は`ell_t*=ell_t/tau_eff`だけを変更する1 variant × 773 HMM runs。
- sigma、missing、Student-t/Huber、downsampling、transition、grid、blendを変更しない。

## 受け入れ基準

- Stage 0 evaluable well率90%以上、fallback率10%以下。
- full vs last-512のSpearman`>=0.70`、median absolute log ratio`<=log(1.5)`、4/5 foldsで同条件を満たす。
- pooled median `tau_eff>=1.25`、upper clip率25%以下。
- fold間median tauのmax/min比`<=1.50`。
- Stage 0 PASSと別承認時だけStage 1を実装する。
- Stage 1はexp281比RMSE`>=0.05 ft`、4/5 folds、1000+・hidden-like・p95非悪化、worst`<=+0.25 ft`。direct promotionにはexp226更新も必要。
- FAIL後のlag/k/clip/temperature/downsampling救済は禁止。

## 次のアクション

Stage 0のKaggle CPU package/push/runが必要なら、実行量を再提示して別承認を得る。
