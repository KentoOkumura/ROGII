# exp255 結果

## 結論

Kaggle CPU train v1は完了しました。3つの固定profileはすべて採用guard不通過です。assertiveはglobal改善を示しましたがworst-well riskを抑えられないため、inference・competition submitには進みません。

## 実行契約

- kernel: `kentookumura/exp255-gated-bounded-selector-readout-train`
- version / id_no: 1 / `127317283`
- runtime: 691.353秒、CPU、internet off
- rows / wells / outer folds: 3,783,989 / 773 / 5
- model config / fold training / booster: 0 / 0 / 0
- parent/control再学習: なし
- selector score: outer-fold role=`valid`のみ、全3,783,989行を一度ずつcover
- selector score 5 foldのdecompressed SHA: 全一致
- truth used in gate: false
- inference / submission: 未実行

## OOF結果

| variant | RMSE | base差 | 改善fold | worst-well差 | guard |
| --- | ---: | ---: | ---: | ---: | --- |
| exp238 add-only base | 7.936690 | 0.000000 | - | - | baseline |
| hard top1 diagnostic | 8.512262 | +0.575572 | - | - | 不採用 |
| conservative | 7.938384 | +0.001694 | 2/5 | +1.163183 | fail |
| balanced | 7.929965 | -0.006725 | 2/5 | +2.167282 | fail |
| assertive | 7.877990 | -0.058700 | 3/5 | +3.151245 | fail |

assertiveのscope差はnear `-0.038930`、1000+ `-0.066824`、hidden-like spatial `-0.189661`、typewell-purged `-0.191090`です。scope平均では一貫して改善しました。

一方、assertiveで補正された604 wellsは324改善 / 280悪化で、106 wellsが`+0.25 ft`を超えて悪化しました。最大は`d7ba4f9d`の`+3.151245`です。global gainは少数のwell-tail riskを相殺して得られており、安全なdirect readoutとは判定できません。

## 解釈

- selector top1方向へ直接動かす信号自体にはglobal headroomがあります。
- ただしgain/marginとwell内candidate/direction consistencyだけでは、悪化wellを識別できません。
- hard top1が大幅悪化するため、clipを外す、alphaを増やす、guardを緩める方向は根拠がありません。
- HMMは全top1の31.87%、exp226は17.57%ですが、この分布だけでは改善・悪化を各candidateへ因果帰属できません。

## 信頼性

結果はtrustworthyです。candidateは全行finite、selector valid-score coverageは100%、5 fold SHA、入力manifest、metrics/by-well/gate/well-gateのSHAが一致し、gateはtruthを受け取りません。Kaggle pull-backした16/16 cell sourceもlocal packageと一致しています。

3.78M行のselected OOF本体はローカルへ同期せず、Kaggle summaryに記録されたdecompressed SHA `2ea22a46c46ef62a0411b170a0f9a81e85664355d5f224f27d73ffd0f929296a`を証拠にします。採用判断に必要なsmall outputsは実験配下へ同期済みです。

## 次アクション

exp255 profileのinference・submitは行いません。再訪する場合は、outer-train wellsだけで悪化riskを学習しouter-valid wellsへ適用するwell-level risk discriminatorを別実験で監査します。exp255 assertiveをfallback付きで使い、worst-well `+0.25 ft`以下を満たせない限り推論へ進めません。
