# exp291_prefix_masked_typewell_gr_multimode_safe_beam

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU version 1完了・固定guard FAIL・branch closed
- CV: known-prefix masked backtest
- Public / Private LB: 対象外
- 親: `exp284_prefix_masked_wrong_mode_branch_recovery_backtest`

## 仮説

same-well self-GRを除外し、eligibleなType Well GR shift局所modeをsafe baseとともに全保持し、
複数checkpointで継続してsafeを上回ったmodeだけへcommitすればfalse switchを抑えられるか検証した。

## 固定契約

- known prefix末尾640行mask、pre-cut score 128行、H128/H256/H512
- 固定13 shift bank、`abs(shift) >= 10 ft`の局所極大全件
- safe-only / top1 / all modes / matched-count shuffleの4 policy
- safe baseは常時保持、same-well self-GR候補は0
- target-free tableをcontent SHA凍結後にだけpost-cut truthへ接続
- active contract 1、LightGBM 0、trained fold 0、booster 0、HMM/PF再生成0

## 検証方針

exp226保存GroupKFoldのouter-valid wellごとにknown prefix末尾640行をmaskし、target-free tableの
content SHA凍結後だけtruthを接続した。technical、pairwise、safe/top1改善、false switch、H512持続性、
matched-count shuffle超えを事前固定guardで判定し、1条件でもFAILなら救済なしで閉じる。

## 結果

| メトリック | 値 |
| --- | ---: |
| eligible wells | 766 |
| H256 safe-only RMSE | 4.827483 |
| H256 top1 RMSE | 18.713110 |
| H256 all-mode RMSE | 22.199818 |
| H256 matched-shuffle RMSE | 17.360718 |
| all-mode gain vs safe | -17.372335 ft |
| false switch | 34.9462% |
| pooled AUC | 0.672737 |
| pooled balanced accuracy | 0.576907 |

technical guardは全PASSしたが、pairwise安定性、safe改善、top1超え、false switch、
matched shuffle超えをFAILした。全体guardはFAILで、`close_without_parameter_rescue` とする。

## 所見

pooled AUC 0.672737だけでは、truth上で良いalternativeが1.2128%しかない不均衡下の安全なcommitには
不十分だった。all-modeはsafe、top1、score-blind shuffleの全てに負けており、候補数やcheckpointの
微調整を支持する証拠はない。

## 実装・実行

compact self-contained trainをcanonical notebookへ採用し、専用9 tests、repository 260 tests、
Ruff、py_compile、Jupytext round-trip、strict validationを通過後、Kaggle CPU version 1を実行した。
runtimeは6,805.497秒。kernelは
`kentookumura/exp291-typewell-multimode-safe-beam-backtest-train`（id_no `127882960`）。

canonical inferenceは未採用で、decoder、prediction、submissionは生成していない。
同じbacktest truthでparameter rescueは行わない。

## 次

exp291 branchをclosedのまま維持し、同じtruthでparameter rescueを行わない。
既存の独立した高優先backlogを優先する。
