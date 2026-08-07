# タスクリスト

## 変更点

exp282のhard donor transferを再調整せず、self-GR matchを最大3本のalternative mode proposalに限定し、
baseを保持した未来256行typewell evidence readoutへ置き換えた。

## 未着手

- なし。

## ブロック中

- selected H256 gain、5/5 nonregression、false-switch guardがFAILしたため、同一仮説の救済grid、
  decoder接続、inference、submissionを閉鎖する。
- exp284は別のユーザー明示overrideでstandalone実行されたが、本実験からscientific promotionを
  付与せず、triggered multibranch decoderへ接続しない。

## 完了

- 未使用の実験番号`exp283`を確認した。
- steeringを作成し、K=3、primary H=256、4 event strata、proposal bank、future evidence、guardを固定した。
- `experiments/exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout/`をtemplateから作成した。
- planned config / README / SESSION_NOTES / result / metricsを設計確定状態へ更新した。
- backlogとexperiment summaryへ設計済み未実装として登録した。
- compact self-contained Jupytext train source/notebookを10章・23 cellsで実装した。
- fail-closed inference source/notebookを3章・8 cellsで実装した。
- event / proposal / evidence / truthの4境界、stable shuffle、H128/256/512、metrics/guardを実装した。
- 専用合成test 7件、関連exp280/282/283 test 19件、ruff、`py_compile`、Jupytext round-tripを通した。
- strict `make validate-exp EXP=exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout`を通した。
- ユーザー承認後にcompact sourceを正規train/inference notebookへ採用し、canonical CPU packageを作成した。
- 長いfull-name slugの`SaveKernel 400`を未作成確認後、意味を保持した49文字slugへ短縮してversion 1を開始した。
- version 1のexp226/exp263 fold-label contract mismatchを評価前technical failureとして修正し、専用testを8件へ増やした。
- 同じkernel idのversion 2を1,331.408秒で完走し、3,783,989 rows / 773 wells、4,397 eventsを監査した。
- technical guard全PASS、proposal lift `+0.033204`とAUC 5/5 PASSを確認した。
- selected gain `-6.384973 ft`、nonregressing fold 0/5、false switch `55.5647%`でscientific FAILを確定した。
- fold / scope / by-well / SHA確認のためoutputを`/tmp`へ取得し、metric CSV 8件と5 gzip生成物を照合した。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`README.md`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`へnegative resultとbranch closeを記録した。

## 次のアクション

exp283はnegative diagnosticとして完了。追加実装は行わず、独立routeの既存backlogを優先する。
