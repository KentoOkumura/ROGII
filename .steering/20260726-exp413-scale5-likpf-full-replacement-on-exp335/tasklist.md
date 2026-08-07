# タスクリスト

## TODO

- [ ] 正規Notebook採用は別承認を得てから行う。

## 進行中

- なし。

## ブロック中

- 正規Notebook編集と外部提出は未承認。別名current-test NotebookのKaggle
  version追加とsubmission output生成だけが承認済み。

## 完了

- [x] `exp413_scale5_likpf_full_replacement_on_exp335`を採番した。
- [x] steering requirements/design/tasklistを作成した。
- [x] 親、単一変更、5 changed / 7 unchanged candidate、feature graphを固定した。
- [x] 40 CPU + 20 CPU + 15 GPU = 75 boosters、control再学習0を固定した。
- [x] primary CV/scope gateとtail report-only方針を固定した。
- [x] `docs/06_reproducibility.md`に基づく再現性設計を記入した。
- [x] 実験scaffoldを作成し、placeholder Notebookのまま実装境界を閉じた。
- [x] 親exp335/exp264のNotebook構成を比較し、別名Jupytext percent形式の
  train候補と`.ipynb`を作成した。
- [x] Stage 0のfrozen prediction SHA、strict row join、5 changed /
  7 unchanged candidate、formula、transitive lineage、stale mean拒否を実装した。
- [x] rebuilt clean273 / selector88 / compact74 / signed23 / final370のschema /
  content SHA manifestを実装した。
- [x] Stage C 40 CPU、Stage S 20 CPU、Stage D 15 GPUを段階別authorization /
  run flag付きで実装した。
- [x] dedicated test 8件、親回帰test 25件、Jupytext round-trip、py_compile、
  Ruff F821/F401/F811、strict experiment validationをPASSした。
- [x] Stage 0 push前にactive variant 1、fold partition 5、CPU/GPU booster 0、
  control再学習0、PF/HMM/Beam 0、metadata/bootstrap configを確認した。
- [x] Stage 0 Kaggle version 3をCOMPLETE / technical PASSした。3,783,989 rows /
  773 wells、5 changed / 7 unchanged、formula / unchanged / old-mean parity max差0。
- [x] Stage 0 outputを取得し、preflight / semantic manifest / 5 fold partitionの
  file/content SHAを記録してlocal verifierをPASSした。
- [x] Stage C version 3を1 variant × 2 objectives × outer 5 × inner 4 =
  40 CPU boostersで完了し、40 models / 25 partitions /
  18,919,945 compact rows / 45,407,868 score rowsを確認した。
- [x] Stage C score guardとleakage auditをPASSし、model / compact /
  candidate-score / lineage / reproducibilityのSHAを記録・照合した。
- [x] Stage S push前に1 variant × 1 objective × outer 5 × inner 4 =
  20 CPU boosters、control再学習0、Stage D 0、metadata/bootstrap configを
  確認した。
- [x] Stage S version 1を20/20 models、25 signed compact partitions、
  18,919,945 rows、45,407,868 score rowsで完了し、technical / score gateを
  PASSした。
- [x] Stage S model / compact / score / lineage / reproducibilityのSHAを
  記録・照合した。
- [x] Stage D version 2を15/15 GPU modelsで完了し、saved exp335比
  `0.261304961 ft`改善、5/5 folds nonworse、全固定scope改善でprimary gateを
  PASSした。model/OOF/feature/reproducibility SHAを記録・照合した。
- [x] 2026-07-29のユーザー依頼により、current-test推論実装、
  Kaggle package/push/run、予測監査生成物の取得を承認範囲として記録した。
- [x] 正規placeholderを変更せず、scale5 semantic slot全面置換、保存済み
  40/20/15 models、学習0、submission生成なしの別名Jupytext推論Notebookと
  CPU/private/internet-off packageを作成し、静的検証をPASSした。
- [x] current-test CPU inference version 2をCOMPLETEし、14,151 rows / 3 wells、
  40/20/15 saved models、row/order/finite、feature 273/74/23、prediction /
  decompressed content SHAを検証した。このKaggle run自体はprediction-only。
- [x] 後続承認後、version 2の取得済みpredictionからローカル事前検証用
  `id,tvt` CSVを作り、14,151 rows、ID順、重複、missing、NaN/inf、source値を
  PASSした。ただしこれはKaggle Notebook outputではなく提出物としては不十分。
- [x] 同じcurrent-test full inference NotebookをKaggle version 3として実行し、
  `/kaggle/working/submission.csv`をNotebook outputとして生成した。
- [x] version 3 outputを取得し、14,151 rows、`id,tvt`、sample ID順、finite、
  source prediction exact parity、SHAのsubmit-checkをPASSした。外部提出は未実行。
- [x] ユーザー実施code submission ref `55078306`のraw APIを確認し、
  hidden rerun unhandled error / scoreなしを記録した。hidden成功済みexp335との差分から
  exp413固有の公開test row / well hard assertを原因として特定した。
- [x] 公開test 14,151 rows / 3 wellsのruntime hard assertを、sample由来の
  dynamic row / ID / nonempty-well contractへ置換した。
- [x] 同じkernel IDのversion 4をKaggle CPUで完了し、version 3 prediction /
  submission exact parity、submit-check、hidden互換静的監査をPASSした。
- [x] ユーザー実施のversion 4 code submission ref `55080377`が
  `SubmissionStatus.COMPLETE`、Public LB `7.201`で完了したことをKaggle CLIで
  確認した。exp335 `7.517`比`-0.316`でML Public-LB referenceを更新した。
