# exp029_public_sel15_pf_oof_feature_generation 結果

## 仮説

公開 sel15 PF/Beam replay の train well の途中以降を隠した疑似 test prediction と confidence diagnostics は、後続の selector / meta-stack で exp026 self-route 基準 と組み合わせる特徴として使える可能性がある。

## 設定

- 親: `exp027_public_replay_needless090_sel15_spread3`
- 検証: train well を cutoff で train well の途中以降を隠した疑似 test 化し、cutoff 以降だけを feature/diagnostic rows として保存
- メトリック: RMSE diagnostics only; selector/meta model CV は未実施
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Local smoke rows | 1847 |
| Local smoke PF RMSE diagnostic | 8.776856 |
| Local smoke last-anchor RMSE diagnostic | 10.200872 |
| Local smoke beam RMSE diagnostic | 10.532402 |
| Kaggle smoke rows | 43542 |
| Kaggle smoke PF RMSE diagnostic | 10.381203 |
| Kaggle smoke last-anchor RMSE diagnostic | 16.266463 |
| Kaggle smoke beam RMSE diagnostic | 15.154074 |
| All-well cutoff 0.65 rows | 1782279 |
| All-well cutoff 0.65 PF RMSE diagnostic | 15.172636 |
| All-well cutoff 0.65 last-anchor RMSE diagnostic | 18.284054 |
| All-well cutoff 0.65 beam RMSE diagnostic | 18.122632 |

## 解釈

実装、local smoke、Kaggle smoke、all-well cutoff 0.65 artifact generation が完了。All-well run は 773 wells / cutoff 0.65 / 16 seeds / 250 particles で 1,782,279 rows を生成し、PF diagnostic は last-anchor と beam を上回った。ただし `reference_oof_rows=0` のため exp026 OOF 差分列は未接続であり、selector / meta-stack 前に exp026 OOF と結合する必要がある。

## 次

All-well summary で PF が hold より悪い well の条件を分析し、次の `public_sel15_pf_candidate_selector` または exp026 OOF join を実装する。
