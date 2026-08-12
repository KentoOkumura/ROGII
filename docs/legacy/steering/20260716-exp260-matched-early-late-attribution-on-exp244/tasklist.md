# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- steering docs作成。
- 再現性設計を`design.md`に記入。
- 2026-07-16、ユーザーが30 boosters / control再学習0のmatched attributionへ進むことを明示承認。
- exp260実験ディレクトリ作成。
- exp244 cacheを1回streamし、direction maskで2 variantを学習するJupytext notebookを実装。
- frozen exp218 / exp244 mixed OOF identityとSHA guardを実装。
- variant別metrics / by-well / fold / hidden-like / model / prediction SHA保存を実装。
- Jupytext変換・test、py_compile、ruff F821、strict experiment validationをpass。
- Kaggle metadataとbootstrap内config/helper/SHA、7 kernel sources、GPU/internet設定を確認。
- 2 variants / 3 configs / 5 folds / 30 boosters、control再学習0を`SESSION_NOTES.md`へ記録。
- canonical GPU train `kentookumura/exp260-matched-early-late-exp244-train` v1をpush。
- pull metadata id_no `127431158`、machine shape `Gpu`、7 kernel sourcesを確認。
- Kaggle GPU train v1で30 boostersを完走し、status `COMPLETE`と例外なしを確認。
- logsからvariant別OOF、stress surface、guard、runtime、model/prediction SHAを記録。
- metrics / training metrics / by-well / summary / logだけをselective downloadし、SHAを照合。
- early-only / late-onlyとも不採用、late独立補償false、inference / submissionなしでbranchを終了。
- `experiment_summary.md`と`KAGGLE_DIRECTION.md`を更新。
