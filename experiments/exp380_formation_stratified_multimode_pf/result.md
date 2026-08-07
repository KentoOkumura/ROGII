# exp380_formation_stratified_multimode_pf 結果

## 仮説

mode最低粒子数を保証するPFはformation-relative仮説の早期消失を防ぐ。

## 設定

- 親: exp271 / PF実装: exp072 / 候補源: exp378
- 検証: outer 5-fold、Stage 0 seed0、Stage 1 mean4
- メトリック: RMSE、ESS、mode survival、novelty
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: stable SHA per well/mode/seed
- kernel version: 未実行
- feature content SHA: 未生成
- model SHA / manifest SHA: 未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

本実験自体は未実装・未実行。前提exp377 v2はStage 0をPASSしたが、
formation-relative median6 pathはdirectより`22.676107 ft`悪化し、改善foldは`0/5`、
個別6面も全悪化した。exp378を開かないため、stratified PFを評価しない。

## 次

現設計を閉じ、773 / 3,092 PF runsを開始しない。
