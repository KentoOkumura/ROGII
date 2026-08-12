# タスクリスト

## 実装中

- なし。

## ブロック中

- なし。Stage A FAILによりStage B、推論、提出はclose。

## 完了

- exp393として採番し、exp347を変更しない独立practical numerical equivalence auditとしてsteeringを作成した。
- scalar/batched差がfloat32 reduction/layout由来かを分離する比較modeを設計した。
- posterior mean TVT、MAP、loss/gradient/update、padding/finiteのpractical gateを事前固定した。
- Stage 0実行量をaudit 1 / fixed16 / temporary model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・control再学習0へ固定した。
- 再現性、SHA、Kaggle bootstrap、GPU quota、no-rescue、段階承認を設計した。
- exp347 compact self-contained trainを構成参照に、exp393 Stage 0 numerical auditを別名Jupytext percent sourceとして実装した。
- scalar FP32、batched FP32 batch 1/4、先頭4-window scalar FP64診断を実装した。
- truth読込前のunary 1回生成/freeze、posterior mean TVT、MAP、total variation、loss/partition/gradient/AdamW update、runtime/memory、SHA reportを実装した。
- compact self-contained trainとfail-closed inferenceを正規Notebookへ採用した。
- Jupytext、py_compile、Ruff F821、専用testを通した。
- Kaggle T4 Stage 0 version 2を完了し、13 gate中10 PASS / 3 FAILで
  `fail_close_without_threshold_dtype_batch_padding_or_kernel_rescue`と判定した。
- ユーザーが3 FAILを承知してStage Aへ進む明示overrideを行った。exp347とStage 0
  FAILは維持し、Stage A science gateも変更しない。
- user overrideをfail-closed guardとしてconfigと正規train Notebookへ追加した。
- exp347のStage A training/freeze/readout/orchestrationをexp393へ追加し、専用test、
  Jupytext、py_compile、Ruff、strict validationを通した。
- package内config、T4、internet off、3 kernel sources、実行量を確認し、version 4で
  Stage A fold 0を完了した。
- Stage Aはreal RMSE `22.866144493 ft` vs exp209 `12.671086935 ft`、well p95
  `43.017462701 ft` vs `26.301518476 ft`、worst-well regression
  `75.227871352 ft`の3 checksをFAILし、Stage Bなしでbranch closeした。
