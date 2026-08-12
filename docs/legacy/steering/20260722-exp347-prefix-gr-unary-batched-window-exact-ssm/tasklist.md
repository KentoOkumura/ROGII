# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage A/B/C、推論、提出: Stage 0 technical parity FAILによりclosed。

## 完了

- exp347として採番し、steeringを作成した。
- exp332を再開せず、batched exact DPを独立実験として系譜固定した。
- batch size 4、実効batch parity、Stage 0 technical/compute gate、実行量、failure policyを設計した。
- 再現性設計と禁止事項を記録した。
- 実験scaffoldを作成し、design-only状態へ固定した。
- ユーザーの実装承認を記録し、exp332 compact self-contained trainを構成参照にbatched DP候補を別名Jupytext percent sourceとして実装した。
- fixed 4-window scalar/batch loss/partition/posterior/gradient/AdamW update parityとpadding mask contractをStage 0 reportへ実装した。
- trainを4-window/1-stepへ変更し、full-well 3-control decodeとStage A freeze-first decodeをstable length順4-well batchへ変更した。
- fail-closed compact inference候補と専用testを追加した。
- compact train/inference候補のJupytext変換、`--test`、py_compile、Ruff、専用pytest、strict experiment validationを完了した。
- ユーザーの固定16-window Stage 0実行承認と実行量ガードを記録した。
- compact self-contained train候補をcanonical train Notebookへ採用した。canonical inferenceは変更していない。
- canonical kernel packageをstrict生成し、T4 version 1をpushした。
- Kaggle T4固定16-window Stage 0 version 1を完了し、report、window/batch/boundary/padding manifest、scalar parity、runtime/memory、SHAを取得・検証した。
- 計算gateは保守的`5.108737 h`、speedup`2.574244x`、peak`5.928168 GB`でPASSしたが、posterior parity `1.4662743e-5 > 1e-6`で総合gateをFAILした。
- 事前failure policyどおりbranchを閉じ、Stage A/B/C、推論、提出、同一exp内の救済を停止した。
