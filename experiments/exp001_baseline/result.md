# exp001_baseline 結果

## 仮説

`TVT_input` 既知 prefix の最後の値は、ROGII の hidden tail 予測における強い null model になる。まずこれを leak-safe な比較基準として固定する。

## 設定

- 親: なし
- 検証: `well_id` GroupKFold、`TVT_input` NaN 行のみ評価
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV OOF RMSE (`last_anchor`) | 15.909853 |
| Mean fold RMSE (`last_anchor`) | 15.894391 |
| CV OOF RMSE (`recent_linear`, reference) | 41.022355 |
| Public LB | 15.883 |
| Private LB | - |

## 解釈

`last_anchor` の OOF RMSE 15.909853 は、公開 notebook 調査に記録されていた last-anchor baseline 15.91 と一致する。実装と評価 mask は妥当な可能性が高い。

2026-05-31 に Kaggle inference notebook から提出し、public LB は 15.883 だった。CV 15.909853 と近く、baseline の検証と公開 LB はおおむね整合している。

一方、recent slope の単純外挿は大きく悪化した。tail が長いため小さな slope bias が蓄積しやすく、次は raw TVT の外挿ではなく `TVT - last_anchor_tvt` の drift target を学習する方針が妥当。

## 次

`exp002_drift_minimal` で prefix / GR / trajectory features を使う drift target baseline を作る。formation surface と NCC は fold-safe にできる設計を切ってから追加する。
