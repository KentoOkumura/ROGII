# exp382_formation_physics_candidate_addonly_on_exp335 結果

## 仮説

exp378物理候補の固定20特徴はexp335の370特徴に相補的である。

## 設定

- 親: exp335、物理候補源: exp378
- 検証: outer5×inner4 strict nested
- メトリック: RMSE、fold/config/scope/well-tail差分
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: false
- seed policy: fixed global seed + nested manifest
- kernel version: 未実行
- feature content SHA: 未生成
- model SHA / manifest SHA: 未生成
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

本実験自体は未実装・未実行。前提exp377 v2はStage 0をPASSしたが、
formation-relative median6 pathはdirectより`22.676107 ft`悪化し、個別6面も全悪化した。
exp378の候補artifactとnovelty evidenceを作らないため、add-only特徴を作らない。

## 次

現設計を閉じ、strict-nested特徴生成と15 GPU boostersを開始しない。
