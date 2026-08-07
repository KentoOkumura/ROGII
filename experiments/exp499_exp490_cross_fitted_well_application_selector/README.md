# exp499_exp490_cross_fitted_well_application_selector

## 状態

- ルート: `ensemble`
- 状態: `completed_fail_closed`
- cross-fitted policy RMSE: `8.514310626`
- always-exp490 RMSE: `8.480155260`
- Public / Private LB: なし
- 親実験: `exp490_geometry_centered_mean_reverting_offset_hmm`
- fallback: 保存済み`exp357_exp226_huber_emission_independent_audit`

## 仮説

exp490は全体RMSEを1.257 ft改善した一方、324/773 wellsで悪化した。unknown suffixの
正解を使わない候補間不一致、posterior状態、visible-prefix物理量から、exp490のsigned
benefitを予測し、安全に適用／fallbackを選べる。

## 変更点

- 保存済み候補予測から32個のtarget-free well特徴をfreezeした。
- signed MSE benefitをouter 5 / inner 4のstrict-nested selectorで予測した。
- exp490を常用する基準と、cross-fittedにexp357へfallbackするpolicyを比較した。
- PF/HMM、候補予測、親control、GPU学習は再実行していない。

## 検証方針

- outer fold: exp490保存済み5 folds
- group: well
- model selection: outer-train内inner 4-foldだけ
- apply rule: predicted benefit > 0
- leakage: feature content/SHA freeze後だけfoldとby-well outcomeを読んだ
- model: weighted Ridge / shallow HGB / always-exp490 safeguard

## 実行

- Kaggle private CPU version 2、id_no `129362815`
- 3,783,989 rows / 773 wells / target-free 32 features
- outer 5 × inner 4、45/45 CPU model fits
- LightGBM / PF / Beam / HMM / GPU / control retraining: すべて0
- technical checks: 全PASS

## 結果

| 指標 | 値 |
| --- | ---: |
| exp357 parent RMSE | 9.737195157 |
| always-exp490 RMSE | 8.480155260 |
| cross-fitted policy RMSE | 8.514310626 |
| policy gain vs always-exp490 | -0.034155367 ft |
| report-only oracle RMSE | 6.560582422 |
| pooled beneficial-well AUC | 0.521151 |
| pooled Spearman | 0.122250 |
| AUC >= 0.55 folds | 1 / 5 |
| exp490適用率 | 92.626%（716/773 wells） |
| applied beneficial precision | 58.101% |
| selected-minus-parent p95 / worst | +7.098191 / +49.602560 ft |

## 所見

単一特徴`mean abs(exp357-exp226)`にはAUC 0.5919、正方向5/5 foldsの弱いsignalが
残った。しかしstrict-nested selector scoreはAUC 0.521に留まり、predictability gateと
safe-router gateをともにFAILした。4 outer foldsはalways-exp490を選び、唯一HGBを
選んだfold 1は8.659383から8.822361へ悪化した。未知wellへの安全な適用判断としては
使えない。

## リスク / 注意

- AUC 0.59の単一特徴は原因説明・ranking候補であって、安全なhard routerの根拠ではない。
- 同じOOFでthreshold、特徴、modelを追加して救済しない。
- exp490のterminal closeを維持し、inference / submissionは作らない。

## 次

本selector branchは終了する。exp500の固定PF機構移植は別仮説・別承認のまま維持し、
exp499の弱いsignalをadaptive gateとして持ち込まない。

