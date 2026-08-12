# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- 親exp218、再現性ガード、最近のcompact notebook実装を確認した。
- 再現性設計を `design.md` に記入した。
- Jupytext train/inference sourceと正規notebookを実装した。
- 保存済みexp218 booster control推論、config差分assert、CV、stress、blend、SHA生成物を実装した。
- config、SESSION_NOTES、result、metrics、README、experiment_summary、KAGGLE_DIRECTIONを更新した。
- Jupytext test、py_compile、ruff F821、strict experiment validation、template validationを完了した。
- canonical private GPU train packageをprepareし、metadataとbootstrap内容を確認した。Kaggle pushはしていない。
- ユーザー承認を受領し、`full_family`（1 variant、3 configs、5 folds、15 boosters、control再学習なし）をconfigとSESSION_NOTESへ記録した。
- 単一canonical train notebookをKaggleへpushし、version 1を完走した。
- logsからplan/config/fold/CV/guard/生成物pathを記録した。
- 評価用CSV/JSONを取得し、380-feature schema、15-model manifest、OOF/feature/model/summary SHAを記録した。
- 全guard failを確認し、回帰variantのinference / submitを不採用とした。
