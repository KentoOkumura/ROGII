# exp336_exp287_formation_tail_attribution_readout 結果

## 状態

Kaggle private CPU version 2を完了し、固定判定は`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`となった。事前登録した6 familyはすべてAND gateを不通過で、formation attribution枝を閉じる。モデル、予測、推論、提出は生成していない。

## 実行

- kernel: `kentookumura/exp336-exp287-formtail-attribution-readout-train`
- id_no / completed version: `128221753 / 2`
- runtime: readout本体`92.458 sec`、Notebook全体`102.406 sec`
- runtime: Kaggle CPU、single worker、BLAS thread 1、GPU/internet off
- 実行量: 6 families / model 0 / LightGBM config 0 / trained fold 0 / booster 0 / control再学習0
- version 1は、同一SHAのhidden-like assignmentが3 pathへ複製されていたためfamily評価前にtechnical ERRORとなった。expected SHA一致copyをconfig pattern順で決定的に選ぶresolverだけを修正し、科学契約、family、boundary、gateは変更していない。

## 判定

| Family | Global Q4−Q1 mean | Median | 正方向fold | Hidden-like正方向 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| plane reference distance | +0.081817 ft | +0.080864 ft | 3/5 | 0/2 | FAIL |
| dense reference distance | +0.350471 ft | +0.314590 ft | 5/5 | 0/2 | FAIL |
| dense neighbor uncertainty | +0.151014 ft | +0.016836 ft | 5/5 | 1/2 | FAIL |
| plane–dense disagreement | +0.229596 ft | +0.128755 ft | 4/5 | 0/2 | FAIL |
| formation consensus spread | +0.014517 ft | +0.080344 ft | 4/5 | 2/2 | FAIL |
| known-prefix formation calibration error | +0.031585 ft | -0.002890 ft | 4/5 | 2/2 | FAIL |

全familyでQ1/Q4は`194/193 wells`、strict quartile edge、global/fold/hidden-like coverage、error非依存境界を通過した。したがって不通過はtechnical欠損ではなく、事前登録した効果量・方向再現性の不足による。

最もglobal evidenceが強かった`dense_reference_distance`は効果量`+0.350471 ft`、median正、5/5 folds正まで通過したが、hidden-like spatial / typewell-purgedのQ4−Q1が`-0.019368 / -0.037255 ft`と両方逆方向だった。`formation_consensus_spread`と`known-prefix formation calibration error`はhidden-like 2面を通過したが、global効果量が`+0.25 ft`に届かなかった。

## OOF report-only

| メトリック | 値 |
| --- | ---: |
| exp287 pooled RMSE | 8.136708220 |
| corrected exp264 pooled RMSE | 8.460811238 |
| pooled delta | -0.324103017 ft |
| well等重みdelta mean | -0.169683553 ft |
| well等重みdelta median | -0.065122300 ft |
| exp287が+1/+3/+5 ft悪化したwell | 80 / 17 / 4 |

この`80/17/4`は本readoutがexp287とcorrected exp264のwell RMSEを直接再計算したreport-only値であり、親実験のclean-control基準の悪化well数とは定義が異なる。

## Technical / leakage audit

- exp287 outer-valid formation cache 5 partition、3,783,989 rows、773 wellsをexactly onceで構成した。
- formation非finite値0、plane/dense reference availabilityは全773 wells、6 familyすべてstrict edge eligibleだった。
- Stage Aで開いたraw value列は`MD/X/Y/Z/TVT_input`だけで、forbidden value列は0件だった。
- freeze manifest SHAをStage BのOOF open前後で照合した。
- exp287/corrected-exp264 OOFのID/well/foldは完全一致し、actual TVT max absolute differenceは`0.0 ft`だった。
- 11成果物が揃い、reproducibility manifestに記録された10子artifact SHAを実ファイルと照合して全一致した。

## 再現性

- Stage A freeze manifest SHA: `e65a9924c11f77008d1574070f71b6cf2d099993e8510eeaf7cc285c5d54979f`
- target-free attributes SHA: `a53a537db8eb9416ef4b83e6529d11d8b10233f02926939c349bd679d09f03aa`
- attribution decision SHA: `06b7bfd64405b8f330ac818efe59ca51c9b1b13babc5aff440d72447eea9f99a`
- scientific contract SHA: `b2c8e40c4912abd29277fae96b462d22aaf9826b0a3d4a799902b0f640ee3328`
- seed policy: RNGなし、well/id/familyを固定順、NumPy linear quantile。
- model/prediction/submission SHA: 非該当。

## 結論

exp287のformation add-only global gainにwell-level tailが伴うこと自体は再確認されたが、6つのtarget-free formation reliability familyのいずれにも、global・fold・hidden-like 2面を同時に満たす安定した事前riskはなかった。よって同じOOFでfamily、threshold、weight、clip、shrink、gateを追加探索せず、このformation attribution救済枝を閉じる。exp287/exp334はtrain-side昇格せず、exp336からinferenceやsubmissionへ進まない。
