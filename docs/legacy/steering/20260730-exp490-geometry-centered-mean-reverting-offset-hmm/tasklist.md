# タスクリスト

## TODO

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 2026-07-30: exp490を新規採番し、科学的親をexp357、routeを`pf_beam`に固定した。
- 2026-07-30: geometry-centered mean reversionの式、half-life、区間境界規約を固定した。
- 2026-07-30: exp357からの単一変更と、固定するHMM/Huber条件を明記した。
- 2026-07-30: Stage 0 fixed32とStage 1 full OOFの実行量・gate・禁止事項を固定した。
- 2026-07-30: `docs/06_reproducibility.md`に沿うSHA・truth-late・実行環境契約を記録した。
- 2026-07-30: 実験ディレクトリ、バックログ、実験一覧へdesign-onlyとして登録した。
- 2026-07-30: ユーザー承認に基づきStage 0 fixed32だけを実装した。
- 2026-07-30: Jupytext percent形式のcompact self-contained train candidateを作り、
  既存の正規placeholder notebookを上書きしなかった。
- 2026-07-30: K16境界、`L_k`、`rho_t`、destination-row ownershipのunit testを作成した。
- 2026-07-30: zero-state、segment half-life、finite、posterior normalization、
  truth-before-freeze sentinelを実装した。
- 2026-07-30: Stage 0予定を1 variant × 32 wells、control再実行0、
  model / booster / GPU 0として再確認した。
- 2026-07-30: py_compile、F821、Jupytext test、6件の契約testをPASSした。
- 2026-07-30: ユーザー承認に基づきcompact candidateをcanonical trainへ採用した。
- 2026-07-30: private / CPU / internet off / run-on-pushのstrict packageを作成し、
  metadata、bootstrap config、fixed32 manifest、episode定義のSHA一致を確認した。
- 2026-07-30: 56文字slugのKaggle SaveKernel 400とpull 403を確認し、科学契約を
  変えず44文字canonical slugへ一度だけ短縮した。
- 2026-07-30: Kaggle private CPU version 1（id_no `129180511`）で
  1 variant × 32 HMM well-runsを完了した。
- 2026-07-30: technical 12/13、mechanism 6/7 PASS。full runtime投影
  `51464.889 sec`とmatched-control by-well p95 `+3.118472 ft`をFAILし、
  `stage0_fail_closed`でbranchを閉じた。
- 2026-07-30: logsからfold、gate、runtime、生成物SHAを記録し、Stage 1、
  inference、submissionを無効のまま維持した。
- 2026-07-31: Stage 0 FAILを維持したままfull 773 wellsへ進むユーザー明示承認を
  記録し、stable SHA256 modulo 4の実行分割と0-HMM strict merge契約を固定した。
- 2026-07-31: Kaggle private CPU shard 0--3 version 1を完了し、773 wells / 3,783,989 rows、
  4 predictionのraw/decompressed SHA、summary、well manifestを検証・固定した。
- 2026-07-31: strict merge version 1（id_no `129321382`）を完了。full RMSE
  `8.480155 ft`でexp357から`1.257040 ft`改善、persistent SSEを`41.4100%`削減した。
- 2026-07-31: 14 gate中12 PASS。by-well p95`+7.257814 ft`、worst well
  `+49.602560 ft`をFAILし、`stage_1_failed_close_without_rescue`で完了した。
- 2026-08-01: exp226 `PredictionResult.geop`とfull OOF scientific contract SHAを
  固定したself-contained inference notebookを実装し、正規notebookへ採用した。
- 2026-08-01: py_compile、Ruff F821、Jupytext round-trip、inference契約test 5件を
  PASSした。実行量はexp226 full fit 1、geometry 3 wells、HMM 3 wells、GPU 0。
- 2026-08-01: Kaggle private CPU inference version 1（id_no `129323029`）を完了。
  3 wells / 14,151 rows、technical 13 / 13 PASS、scientific contract SHA一致。
- 2026-08-01: `submission.csv`を取得し、sample header/row/ID順序、有限値、重複、
  SHAを検証。submit-checkはFAIL 0 / WARN 0。competition submitは未実行。
- 2026-08-01: ユーザーが検証済みinference version 1のcompetition submitを明示承認。
- 2026-08-01: submission ref `55163886`を送信。Kaggle APIは`COMPLETE`だがhidden
  再実行が未処理例外となりPublic scoreなし。公開sample SHA / 14,151 rows / 3 wellsの
  固定guardをhidden非互換の原因と診断し、再提出せず失敗記録を確定した。
- 2026-08-02: ユーザーがhidden-dynamic inference version 2の実装修正を承認。
  物理モデル不変、Kaggle push/run/再提出は別承認とする設計を固定した。
- 2026-08-02: compact/canonical inference sourceとnotebookをhidden-dynamicへ修正。
  全test inventory preflight、runtime rows/wells、sample/raw/exp226 identity gateを実装した。
- 2026-08-02: 公開3 wells全件とsynthetic可変2 wells / mismatch reject testを追加。
  scientific contract SHA不変、Kaggle version 2 run approval falseを確認した。
- 2026-08-02: py_compile、Ruff F821、Jupytext、全18契約test、strict validationをPASS。
  929,764-byte strict packageのbootstrap config/source SHA一致を確認した。
- 2026-08-02: ユーザーがcanonical inference version 2のpush/runを明示承認。
  competition submitは未承認のまま分離した。
- 2026-08-02: canonical private CPU inference version 2をCOMPLETE。runtime inventoryを含む
  technical 14 / 14 PASS、output取得、submit-check FAIL 0 / WARN 0、v1/v2 submission
  byte-identicalを確認した。run承認を使用済みとして無効化し、competition submitは行わない。
- 2026-08-02: ユーザー明示承認でcanonical inference version 2をcompetition submit。
  ref `55180208`はhidden再実行を通過して`COMPLETE`、Public LB `9.680`。exp226 direct
  `9.837`は0.157改善したが、direct exact HMM `9.063`より0.617悪いため、物理route
  anchorへ昇格せずStage 1 fail-closeとterminal closeを維持した。
