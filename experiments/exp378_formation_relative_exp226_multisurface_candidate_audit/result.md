# exp378_formation_relative_exp226_multisurface_candidate_audit 結果

## 仮説

formation-relative rateを通した7物理候補はexp226にない有用な変動を持つ。

## 設定

- 親: exp226、前提: exp377
- 検証: outer 5-fold / fixed median primary / fixed12+7 bank
- メトリック: RMSE、oracle gain、unique-best率、候補相関
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: 設計上true、未実行
- seed policy: no RNG
- kernel version: 未実行
- feature content SHA: 未生成
- model SHA / manifest SHA: 対象外
- prediction SHA: 未生成
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

本実験自体は未実装・未実行。前提exp377 v2はStage 0をPASSしてtruth-late評価まで到達したが、
median6 path RMSEがdirect `16.100131`から`38.776238 ft`へ悪化し、改善foldは`0/5`だった。
個別6 formation pathも`39.022186--40.355628 ft`で全てdirectより悪かったため、
7候補化・候補bank監査へ進む科学的根拠がない。

## 次

現設計を閉じ、exp379 / exp380 / exp382へ候補を渡さない。
