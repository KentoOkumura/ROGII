# exp292_typewell_gr_warp_rate_identifiability_audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU完了・`FAIL_CLOSE_NO_RESCUE_GRID`
- primary H256 eligible wells: 29/773（3.7516%）
- AUC real / shuffled / lift: 0.484190 / 0.531181 / -0.046991
- top1 / safe RMSE: 11.938287 / 11.938287
- Public / Private LB: 対象外
- kernel id: `127888550`
- 親実験: `exp268_multi_scale_initial_rate_candidates`
- 設計の正: `.steering/20260719-exp292-typewell-gr-warp-rate-identifiability-audit/`

## 仮説

`exp268` の `tail30 / w32 / w64 / w128 / w256` を固定し、known prefixだけで校正したType Well
forward GRとhorizontal GRの整合性からtruth-best rate candidateをtarget-freeに識別できるかを監査した。

## 実装

- H128/H256/H512、primary H256。
- Gaussian residual / NCC / chain-rule derivative residualの固定等重みscore。
- stable within-well circular-shuffleとalways-tail30 safe control。
- score/eligibility/top1 SHA freeze後にだけtrue TVTをjoin。
- 1 audit variant / 0 configs / 0 trained folds / 0 boosters / 0 HMM/PF regeneration。
- inference、selected row prediction、submissionはdisabled。

## 検証方針

- well単位5-foldでAUC liftとRMSE gainの再現性を読むが、fold内model fitやthreshold fitは行わない。
- primaryはH256だけとし、eligible well/row coverage、real-minus-shuffle AUC、top1対safe RMSE、
  1000+ / hidden-like非悪化を事前guardで一括判定する。
- target-free score/selectionのcontent/schema SHAを凍結してからtrue TVTをjoinする。
- 1つでもguardをFAILした場合は救済gridを行わずbranchを閉じる。

## 結果

technical contractと全SHA guardはPASSした。一方、H256 coverageはwell 3.7516%、row 3.6178%で、
90% guardを大きく下回った。real AUCはshuffleより0.046991低く、正のAUC liftは0/5 foldsだった。
全773 wellsでtop1はtail30 safeのままでRMSE gainは0、改善foldも0/5だった。

1000+とhidden-like 2面は非悪化だが、全wellがsafe fallbackしたためであり、追加GR scoreの価値を示さない。

## 実行入口

- canonical train: `exp292_typewell_gr_warp_rate_identifiability_audit_train.ipynb`
- canonical source: `exp292_typewell_gr_warp_rate_identifiability_audit_compact_selfcontained_train.py`
- inference: 未採用・fail-closedのまま

## 所見

Type Well forward-GR frequency/shape scoreは、固定した厳格な適格条件では広く測定できず、測定可能な
29 wellsでもrandomized controlを上回らなかった。exp268の小さいoracle headroomをtarget-freeに
回収する方法としては不成立である。

## 次

事前登録した停止条件に従ってfrequency-warp rate branchを閉じる。threshold、horizon、calibration、
weight等の救済grid、top1 replacement、raw-test inference、submissionは行わない。
