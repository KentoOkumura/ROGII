# exp354_typewell_group_candidate_family_prior_readout 結果

## 状態

Kaggle private CPU version 1を完了し、固定Stage 0 gateはFAILした。
10 checks中9 PASSだったが、real-minus-shuffle SpearmanだけがFAILしたため、
救済調整、再実行、Stage 1、inference、submissionなしでbranchを閉じる。

- kernel: `kentookumura/exp354-typewell-family-prior-readout-train`
- version / id_no: `1 / 128363177`
- diagnostic runtime: `63.672340 sec`
- runtime: CPU、GPU/internet off

## 仮説

exp293固定candidate familyについて、outer-train Type Well群×family soft error priorが
held-out wellのfamily実績順位を、group-label shuffleを上回って再現するかを判定する。

## 判定予約

- candidate/family/fold parity、全prior finite、fit-valid overlap 0
- held-out group coverage `>=0.90`
- family rank Spearman `>=0.15`、正方向 `>=4/5 folds`
- real minus group-label-shuffle Spearman `>=0.05`
- hidden-like spatial / typewell-purged Spearman非負

## Stage 0結果

| Gate | 値 | 判定 |
| --- | ---: | --- |
| candidate/family/fold parity | true | PASS |
| all prior finite | true | PASS |
| fit-valid well overlap | 0 | PASS |
| truth rows before prior freeze | 0 | PASS |
| held-out group coverage | 0.980595 | PASS |
| real family rank Spearman | 0.325789 | PASS |
| positive folds | 5/5 | PASS |
| hidden-like spatial Spearman | 0.381736 | PASS |
| hidden-like typewell-purged Spearman | 0.376570 | PASS |
| shuffle family rank Spearman | 0.327079 | control |
| real minus shuffle Spearman | -0.001290 | **FAIL** |

real priorのfold別Spearmanは`0.302373--0.342908`で全fold正方向だった。
一方、stable group-label shuffleも`0.304631--0.355529`で同水準となり、
overall差は固定下限`+0.05`に対して`-0.001290`だった。

## 実行契約

- prior variant / negative control: `1 / 1`
- reporting folds: `5`
- model config / trained fold / booster: `0 / 0 / 0`
- candidate/PF/Beam再生成、親control再学習: `0 / 0`
- Stage 1 selector model: `0`。予約40 modelsは未実装・未実行。

## 再現性

- target-free input freeze SHA:
  `b1920c1eb6a855201a91ede193eabdf0fdeead959afb560a455a65c27bb527cd`
- prior schedule freeze SHA:
  `f64975ec2335b8eb5f1bf3e03d8e8a4d3314a9b878219941ded66be583b173f2`
- prior schedule decompressed content SHA:
  `1d0f0073ab5a1f997bde06e1fa7366088d56a21fa507083ac8f39db8590d2585`
- readout decompressed content SHA:
  `4c384076578107fbff22268222dba4559c09227cd6e8d6f2b7ad6c9f8157b7a8`
- well-family error decompressed content SHA:
  `9afa8a83520ef511c3fc721bc2441f8511516433ff675fc2112b03d105d3f8a1`
- model / prediction / submission SHA: 非該当。
- rerun parity: 未実行。fixed negative-control permutationのSHAは保存済み。

## 解釈

real prior自体はfamily順位と相関したが、native Type Well group labelをshuffleしても
相関が落ちなかった。したがって、rank signalはgroup固有差ではなく、主に
outer-train global family base rateやfamily共通の難易度構造を読んでいると解釈する。
Type Well群×family priorをcorrected exp264 selectorへ加える根拠は成立しない。

## 次

- support、family、group定義、rank statisticを同じreadout上で救済探索しない。
- Stage 1の40 selector models、raw-test prior、inference、submissionへ進まない。
- exp353はgroup qualityをexp148 errorと結ぶ別仮説だが、exp354の結果はpromotion根拠にしない。
  実行・判断する場合もexp353自身のfixed real-minus-shuffle gateを正とする。
