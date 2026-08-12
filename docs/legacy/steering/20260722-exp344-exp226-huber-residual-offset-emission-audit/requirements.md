# 要件

## 依頼

- exp226 residual-offset HMMでGaussianをHuber emissionへ置換する案を、Student-tとは別の条件付き実験として設計確定する。
- 実装・実行はまだ行わない。

## 制約

- Route: `pf_beam`。
- 親はStage 0 `exp280_exp226_shift_likelihood_separability_readout`、Stage 1 `exp281_exp226_residual_offset_exact_hmm_transition_probe`。
- dependencyはexp342 Stage 0の確定結果。Student-tが全scope PASSならHuberは冗長として実行せず、Student-tがextreme-residual blockを改善したがoverall margin/rankをFAILした場合だけHuber Stage 0を許可する。Student-tがextreme residualでも不支持ならrobust emission familyを閉じる。
- Huberは標準化残差`z`へ`delta=1.345`、`rho(z)=0.5*z^2` if `|z|<=delta` else `delta*(|z|-0.5*delta)`、`ell=-rho`に固定する。
- sigma、missing、block/shift、exp226 shape、HMM grammar、transition、prior、posterior meanを固定する。
- delta/cap/scale/temperature grid、Student-tとの同時full runは禁止。

## 受け入れ基準

- Stage 0はexp280 Gaussian parity、coverage、shift identityをPASSする。
- HuberがGaussian比pooled MRR/top3を各`>=0.01`、4/5 foldsで改善し、1000+・hidden-like・persistent-offsetを悪化させない。
- real-vs-shuffle gapと`|z|>=3` block regretを改善する。
- Stage 0全gate PASSと別承認時だけStage 1を1 variant × 773 runsで実施する。
- Stage 1 gateはexp342と同じ: exp281比`>=0.05 ft`、4/5 folds、tail/p95非悪化、worst`<=+0.25 ft`、direct promotionはexp226更新。
- FAIL後のdelta/log-cap/scale/weight救済は禁止。

## 次のアクション

exp342 Stage 0のdependency condition成立後にのみ、実装承認を求める。
