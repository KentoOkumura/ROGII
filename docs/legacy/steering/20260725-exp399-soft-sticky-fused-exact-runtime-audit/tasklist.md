# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `docs/06_reproducibility.md`を確認し、RNGなし・stable output order・SHA方針を設計した。
- scientific candidate 1 / fixed16 HMM runs 16 / model・booster・control rerun・GPU各0を固定した。
- exp394 sourceをexp399へrenameし、親生成物load-only契約とStage 0実行量を設定した。
- on-the-fly sparse transition、fused docking、2-well outer parallelを実装した。
- dense small-trellis、親kernel、境界、single/parallel parityの専用testを追加した。
- compact self-contained Jupytext候補を正規train notebookへ変換・検証した。
- Kaggle package metadataとbootstrap内configの整合を確認した。
- fixed16 private CPU auditを実行し、parent parity、`6.168148x` speedup、
  full runtime projection `18,277.265 sec`を記録した。
- fixed16 PASS後に別承認を得て、version 6で773-well full OOFを完了した。
- full OOFは全technical gateをPASSしたが、RMSE `11.395646`でexp263より
  `3.157314 ft`悪化し、promotionを棄却した。
- OOF、branch posterior、schedule、promotion gateを取得し、raw SHAとgzip integrityを確認した。
- 同一OOF rescue、inference、submissionを行わず、run flagをfalseへ戻した。
