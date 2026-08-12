# タスクリスト

## 目的

exp293の設計、compact実装、canonical採用、Kaggle実行、結果記録を分離して管理する。

## 親との差分

exp263のcandidate生成、値、formula、fold、suffixは変更しない。exp293では保存済みdeployable12を読む
SHA付きresolver、truth freeze、row/block/whole-well oracle集約、固定support判定だけを追加する。

## 進行中

- なし。

## ブロック中

- なし。exp293の固定監査は完了した。

## 完了

- backlog、steering、実験scaffold、primary deployable12、oracle粒度、support条件、2/3/4分岐を固定した。
- `docs/06_reproducibility.md`に基づくSHA、RNG、Kaggle bootstrap方針を記録した。
- compact self-contained trainとfail-closed inferenceをJupytext形式で実装した。
- exp263 resolver、SHA/parity guard、block assignment、truth freeze、chunked oracle、readout、support判定を実装した。
- downstream branch parityを含む専用11 testsとrepository 298 testsを通過した。
- compact trainをcanonical trainへ採用し、strict Kaggle CPU packageを監査した。
- version 1のMarkdown package不足を記録し、scientific contractを変えず固定contract SHAへ修正した。
- 同じcanonical kernelのversion 2を完了し、3,783,989 rows / 773 wells / 12 candidatesを監査した。
- H512 oracle RMSE `3.683763`、fold最大`4.117908`、必要回収率`0.471825`でsupport PASSを確定した。
- output SHA manifest 11件のfile/decompressed SHAを再検証し、不一致0を確認した。
- `result.md`、`metrics.json`、`SESSION_NOTES.md`、`experiment_summary.md`、backlogを更新した。
- inference、submission、oracle/selected row predictionは生成しなかった。

## 次

固定分岐どおりStage 4を開始せず、Stage 2 `prefix_calibrated_latent_registration_gr_evidence`だけを
別steering/実験として切り出す。Stage 2実装はユーザーの別承認を待つ。
