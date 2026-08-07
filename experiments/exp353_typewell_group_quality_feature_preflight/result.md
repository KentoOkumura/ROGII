# exp353_typewell_group_quality_feature_preflight 結果

## 状態

Kaggle private CPU version 1（id_no `128362932`）でStage 0を完了した。
8 checks中5 PASSだったが、固定scientific gateはFAILした。Stage 1、inference、submissionは行わない。

## 仮説

Type Well群quality 6列がsoftなML補助特徴として有望かを、exp148 OOF errorとの
fold-safe associationで先に判定する。

## 判定予約

- coverage `>=0.90`、fallback `<=0.10`、全feature finite
- residual sigma対exp148 well RMSE Spearman `>=0.15`、正方向 `>=4/5 folds`
- exp148 RMSEのq4-q1差 `>=0.25 ft`
- real minus group-label shuffle Spearman `>=0.05`
- PASS後のStage 1はexp148比 `>=0.03 ft`、4/5 folds、tail/hidden-like guard

## Stage 0結果

| 指標 | 値 | gate |
| --- | ---: | --- |
| coverage | 0.980595 | PASS (`>=0.90`) |
| fallback | 0.019405 | PASS (`<=0.10`) |
| feature finite | 100% | PASS |
| freeze前outer-valid truth | 0 | PASS |
| residual sigma vs exp148 well-RMSE Spearman | 0.006134 | FAIL (`>=0.15`) |
| 正方向fold | 4/5 | PASS |
| q4-q1 exp148 well-RMSE | +0.202701 ft | FAIL (`>=+0.25`) |
| real minus shuffle Spearman | -0.059166 | FAIL (`>=+0.05`) |

shuffle Spearmanは`0.065301`でreal`0.006134`を上回った。real fold別相関は
`0.044483 / 0.015301 / -0.104195 / 0.021856 / 0.098298`だった。

## 再現性

- deterministic anchor: いいえ。単発version 1でrerun parityなし。
- seed policy: Stage 0 RNGなし、Stage 1はexp148設定継承。
- exp065 membershipとexp148 summary/by-wellをraw SHAで固定した。
- exp148 foldは`TVT_input`欠損3,783,989行のtarget-free row countから再構成する。
- feature manifestを凍結した後だけexp148 by-well OOF errorを開く。
- feature manifest freeze SHA:
  `6a90ee2aa35029c4910e93c7476aa5cff1cce82af0e17fce41d7e34e85e256fe`。
- model / OOF SHA: Stage 0はmodel・prediction各0のため非該当。
- submission SHA: 非該当。

## 解釈

coverageは十分だったが、Type Well群のresidual sigmaはexp148 well errorとほぼ無相関だった。
shuffleがrealを上回ったため、観測した弱いquartile差はnative group固有の品質signalとは扱えない。
exp352で見えた平均transfer gainをsoft quality特徴へ変換しても、Stage 1を正当化する事前根拠にはならなかった。

## 次

同じreadout上で列選択、group/fallback/閾値調整を行わずbranchを閉じる。
Stage 1の15 GPU boosters、raw-test、inference、submissionへ進まない。
