# exp379_formation_transition_mixture_exact_hmm 結果

## 仮説

formation別transition modeをexact HMMで周辺化するとexp209より改善する。

## 設定

- 親: exp209、候補源: exp378
- 検証: Stage 0固定16坑井、合格後outer 5-fold
- メトリック: RMSE、posterior mass、runtime、RSS
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: 設計上true、未実行
- seed policy: no RNG exact HMM
- kernel version: 未実行
- feature content SHA: 未生成
- model SHA / manifest SHA: 未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

本実験自体は未実装・未実行。前提exp377 v2はStage 0をPASSしたが、
formation-relative median6 pathはdirectより`22.676107 ft`悪化し、改善foldは`0/5`、
個別6面も全悪化した。exp378を開かないため、formation transition modeを評価しない。

## 次

現設計を閉じ、16坑井Stage 0と773 HMM runsを開始しない。
