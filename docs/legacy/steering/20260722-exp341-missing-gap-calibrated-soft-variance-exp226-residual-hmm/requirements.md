# 要件

## 依頼

- exp339でouter-fold校正した補間誤差分散を、exp281 residual-offset HMMのraw missing rowだけへsoft varianceとして加える設計を確定する。
- exp339 PASSと別承認まで実装・実行しない。

## 制約

- Route: `pf_beam`。
- shape親は`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`、decoder controlは保存済み`exp281_exp226_residual_offset_exact_hmm_transition_probe`。
- required dependencyは`exp339_missing_gap_pseudomask_uncertainty_readout`の全gate PASSとtable SHA freeze。
- raw observed GR rowは`σ_eff=σ_GR`、original raw missing rowだけ`σ_eff,t^2=σ_GR^2+σ_imp^2(L,d)`とする。
- exp281と同じinterpolated GR値、base sigma、typewell、offset grid、41 rate states、transition、prior、posterior meanを固定する。
- run lengthは64超をexp339の`32--64`binへcapし、anchorなしは同binの最大distance cellへfallbackする。
- 1 scientific variant × 773 HMM runs、control再実行0、model/config/fold/booster/PF/Beam各0。

## 受け入れ基準

- exp339 dependency、table content SHA、fold exclusion、raw missing mask、exp281 control SHAが全一致する。
- exp281 Gaussian control比RMSE`>=0.05 ft`改善、4/5 folds改善。
- 1000+、hidden-like 2面、by-well p95を悪化させず、worst-well regression`<=+0.25 ft`。
- direct promotionにはexp226 RMSE`9.427110`を更新することを追加要求する。
- missing-run binsすべてでfinite coverage100%、observed row weight/variance parity完全一致。
- 1 gateでもFAILならweight/floor/bin/sigma/Student-t/transition/blend救済、inference、submissionへ進まない。

## 次のアクション

exp339が固定dependency gateをFAILしたため、未実装・未実行のまま閉じる。
