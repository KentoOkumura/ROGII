# exp388 exp374 fixed13 dual selector on exp264

exp374の固定`df=4` Student-t absolute-TVT exact-HMM予測を、
corrected exp264 candidate-long dual selectorの13本目へ追加する実験です。

## 状態

Kaggle private CPU version 1を完了し、科学gate FAILでbranchを閉じました。

## 仮説

exp374単独候補は平均RMSEを`0.217809 ft`改善した一方、by-well tail gateに
失敗しています。単独採用は危険でも、既存候補とのdisagreement、posterior std、
HMM log-likelihood、raw-test-safe contextを使うselectorなら、改善する局所だけを
選んでfixed12 hard selectorを改善できる可能性があります。

## 検証方針

- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp374_exp209_student_t_exact_hmm_emission`
- 追加候補: `student_t_exact_hmm`
- 実行: Stage A + Stage C、CPU selector 40本
- control再学習 / GPU / downstream TVT / inference / submission: `0`

fixed12に13本目を追加する以外のselector設定を固定し、pooled/fold/scope/
by-well tailを事前固定gateで評価します。exp374のtail gate失敗は再分類せず、
same-OOF rescueも行いません。

## 所見

Student-t候補はpooled `18.30%`、5/5 foldsで選ばれ、selector score guardも
PASSしました。しかしhard selector RMSEは親fixed12の`8.652532`から
`8.736104`へ`0.083572 ft`悪化し、改善は2/5 foldsでした。

1000+、hidden-like 2面、by-well p95、worst-wellもFAILしたため、
downstream TVT、inference、submissionへ進めません。H512/whole-well oracleには
`0.097299 / 0.073408 ft`の補完性がありますが、現行hard selectorでは安全に
利用できない結果です。

初回実行先はKaggle Notebookです。ローカルNotebook実行は行いません。
