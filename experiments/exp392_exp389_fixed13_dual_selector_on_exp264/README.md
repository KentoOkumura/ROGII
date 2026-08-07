# exp392 exp389 fixed13 dual selector on exp264

exp389の固定`delta=1.345` Huber absolute-TVT exact-HMM予測を、
corrected exp264 candidate-long dual selectorの13本目へ追加した実験です。

## 状態

Kaggle private CPU version 1を完了し、科学gate FAILでbranchを閉じました。

## 仮説

exp389のHuber候補はGaussian control比でoverall RMSEを
`0.085546 ft`改善し、5/5 foldsと全required scopeで改善しました。一方で
by-well tail gateには失敗しています。候補単独では危険でも、既存候補との
disagreement、posterior std、HMM log-likelihood、raw-test-safe contextを使う
selectorなら、改善する局所だけを選べる可能性を検証しました。

## 検証方針

- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp389_exp209_huber_exact_hmm_emission`
- 追加候補: `huber_exact_hmm`
- 実行: Stage A + Stage C、CPU selector 40本
- control再学習 / GPU / downstream TVT / inference / submission: `0`

corrected exp264のfixed12へHuber候補だけを追加しました。科学gateをFAILした
exp388 Student-t候補は併用せず、fixed14にはしていません。primary hard-select
domain以外は変更せず、7候補fixed fallback domainも固定しています。

## 所見

technical / leakage / selector score guardはPASSし、Huberは91,035 rows、
pooled `2.405795%`、5/5 foldsでtop1利用されました。しかしfixed13 hard RMSEは
親fixed12の`8.652532`から`8.769792`へ`0.117260 ft`悪化し、改善は2/5 foldsです。

1000+、hidden-like 2面、by-well p95、worst-wellもFAILしました。worst
`8902c3f6`はHuber top1率0%にもかかわらず`+7.875188 ft`悪化し、利用率と
well deltaのPearson相関も`0.004539`です。追加候補の直接誤選択だけでなく、
候補追加後のselector再学習が既存12候補をrerankingする不安定性を示します。

H512 / whole-well oracle headroomは`0.003663 / 0.010120 ft`と小さく、
downstream TVT、current-test生成、inference、submission、same-OOF救済へ
進めません。
