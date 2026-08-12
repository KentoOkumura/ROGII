# 設計

## アプローチ

`log slope, intercept, log sigma`をglobal/group/wellに分解したStudent-t residual modelをouter-trainでfitする。主variantはexp311の経験結果に合わせ、slope=1/intercept=0を固定しsigmaだけ階層化する。deterministic MAPとLaplace近似でposterior predictiveを得て、held-out suffix NLL/RMSEをplug-in priorと比較する。

## 実験範囲

- 対象: `exp317_hierarchical_bayesian_typewell_gr_calibration`
- Route: `pf_beam`
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 変更: plug-inからpartial poolingへ。
- 固定: fold、群、df=5、MAP/Laplace、identity-affine primary、gate。
- 計算量: primary 1 + diagnostics 3、5 folds、booster/decoder 0。

## 再現性設計

- deterministic optimizer初期値、stable well順、固定収束tol/iterationsを実装時にconfig値として持つ。
- input pair/group manifest、posterior parameter/schema/content SHA、optimizer statusを保存する。
- MCMCを使わないためchain seed/diagnosticは対象外。

## リスクと停止条件

- full affine階層が改善してもprimary sigma-onlyがFAILなら直接採用しない。
- optimizer non-convergenceをsilent fallbackせずtechnical FAILとする。
- HMM emission統合は本expの範囲外で、PASS後も別設計とする。
