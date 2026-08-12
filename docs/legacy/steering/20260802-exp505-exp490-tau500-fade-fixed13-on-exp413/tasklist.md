# タスクリスト

## 設計差分

- raw exp490 slotをtau=500 fade slotへ1対1置換する。
- selector-level tail progression gateを通った場合だけexp413 downstreamへ受け渡す。
- standalone / direct final predictionではなく、物理候補をML表現として利用する。

## 未着手

- なし。Stage C gate FAILによりbranchを終端閉鎖した。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- exp505番号、名前、route、親、比較対象を確定。
- tau=500 / alpha=1のfade式とraw exp490の1対1置換を確定。
- Stage C fixed13 selectorのfold、モデル、実行量、progression gateを確定。
- Stage C PASS後だけ有効なStage D exp413 replacementとGPUコストを確定。
- truth-late、SHA、selection-bias表記、再現性、禁止事項を確定。
- steering、backlog、experiment scaffoldを設計のみの状態で作成。
- train / inference Notebookをmarkdown-only placeholderに固定。
- ユーザーの実装承認を記録し、Stage C compact self-contained train候補を実装。
- alpha=1 / tau=500式、8列allowlist、raw/decompressed SHA、global key、suffix、md_since、
  source fold不使用、truth-lateをコード上で固定。
- exp501と同じ1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU boostersを実装し、
  control / HMM / PF / Beam / Stage D / inference / submissionを実行pathから除外。
- candidate / feature contractと5件のcontract testを追加し、pytest、ruff、py_compile、
  Jupytext test、`make validate-exp`をPASS。
- 親exp501 compactと同じ9章を維持し、正規placeholderを上書きせず別名compact候補を生成。
- ユーザーのStage C実行承認を記録。正規Notebook採用、Kaggle CPU package、
  1 variant / 2 objectives / outer 5 × inner 4 = 40 CPU boostersのpush/runを許可。
- 正規train NotebookとKaggle packageを検証し、version 1（id_no `129519165`）を完走。
- technical checks全PASS、hard OOF `8.243315437`、raw exp501比gain `0.021574771 ft`、
  4/5 folds、固定7 scope、fade利用条件PASSを確認。
- p95縮小`0.000036536 ft < 0.10 ft`、worst縮小`0.173168079 ft < 1.0 ft`をFAILし、
  `FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`で閉鎖。
- result / metrics / README / SESSION_NOTES / experiment summary / directionを実測値で更新。

## 終了後の禁止

- Stage D、GPU、inference、submissionを実装・実行しない。
- tau / alpha / cutoff / threshold / feature / model / gateのsame-OOF救済を行わない。
