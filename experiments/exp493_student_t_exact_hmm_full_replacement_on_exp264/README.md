# exp493_student_t_exact_hmm_full_replacement_on_exp264

## 状態

- ルート: `ensemble`
- 状態: `stage_c_completed_scientific_gate_failed_closed`
- CV: `8.616237400`
- 保存exp264 CV: `8.652531956`
- delta: `-0.036294555 ft`
- decision: `FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- 物理候補親: `exp374_exp209_student_t_exact_hmm_emission`

## 仮説

exp374で平均改善したdf=4 Student-t exact HMMを、候補数を増やさず
exp264 fixed12のGaussian exact-HMM familyへ全面置換すれば、
selectorの平均精度とwell-tail安全性を両立できる可能性がある。

## 実験

exp264の12候補ID・順序・domain・88列feature schemaを維持し、
Gaussian `exact_hmm` semantic slotをexp374 Student-tへ全面置換した。
HMM依存pair 2本と固定3-wayだけを再計算し、4 changed / 8 unchanged、
fixed12のままstrict nested dual selectorを評価した。

## 検証方針

保存exp264をcontrolとして再学習せず、pooled RMSE、fold改善数、near、
1000+、hidden-like 2面、by-well p95、worst-wellを事前固定AND gateで判定する。
candidate parity、truth-late、leakage、selector score guardも必須とする。

## 実行

- Kaggle private CPU version 3
- 1 variant / 2 objectives / outer 5 × inner 4
- version 3: 40/40 selector booster
- version 2との累計: 80 CPU booster
- 保存control再学習 / GPU / downstream / inference / submission: すべて0
- runtime: `5896.184330 sec`

version 1は親config未解決で学習前停止した。version 2は全40 booster後の
feature-importance集計で停止し、成果物を回収できなかった。修正版version 3は
最後までCOMPLETEした。

## 結果

pooled、near、1000+、hidden-like 2面は改善したが、改善foldは3/5。
by-well p95は`+0.540095855 ft`、worst `f6d009f4`は
`+10.472288433 ft`悪化し、固定scientific gateをFAILした。
Student-t依存familyのtop1率は`36.281580%`。

## 所見

候補数増加の交絡を除いてもwell-tail不安定性は残ったため、hard replacement、
weight / threshold / domain / gate救済、downstream、推論、提出なしで閉じる。

## 次の行動

exp493はterminal closeとし、同一OOFでの救済や再実行は行わない。
Gaussian--Student-t disagreementの0-booster feature-only監査は、
独立した必要性とユーザー承認が生じた場合だけ別実験として検討する。

詳細は`result.md`、`metrics.json`、`SESSION_NOTES.md`、
`.steering/20260730-exp493-student-t-exact-hmm-full-replacement-on-exp264/`を参照する。
