# タスクリスト

## TODO

- なし。FAIL_CLOSEのためinference / submissionへ進めない。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- [x] exp413とexp501のlineage、feature count、fold manifest、保存生成物SHAを照合した。
- [x] add-onlyを棄却し、`nested74 -> compact77`の置換契約を固定した。
- [x] final featureを`clean273 + compact77 + signed23 = 373`に固定した。
- [x] control再学習0、selector再学習0、将来の新規GPU booster 15本を固定した。
- [x] exp413継承のpromotion gateとexp501 tail FAILの扱いを固定した。
- [x] 再現性設計を`design.md`と実験configへ記録した。
- [x] backlog、steering、design-only実験scaffoldを作成した。
- [x] exp413 Stage D の compact self-contained構成を参照し、Jupytext percent形式の別名train sourceを作った。
- [x] exp413 clean273 / signed23 と exp501 compact77 の保存生成物resolverを実装した。
- [x] old nested74 block除外、final373、列順、重複0、key/fold parityを検証するfeature-surface contractを実装した。
- [x] treatment 1 × config 3 × fold 5 = 15 GPU boostersだけが有効なcost guardを実装した。
- [x] feature assembly、fold leakage、control SHA、実行本数のtestを追加した。
- [x] Jupytext round-trip、`py_compile`、`ruff --select F821`、strict validationを通した。
- [x] 15 GPU boosters、control/selector再学習0を再掲し、正規train notebook採用・package・run承認を得た。
- [x] 正規train notebookを採用し、strict Kaggle packageのmetadata/bootstrap/input/T4設定を検証した。
- [x] 58文字slugのSaveKernel 400と未作成を確認し、同じexp502内の42文字canonical slugへ短縮した。
- [x] canonical kernel version 1をT4指定でpushし、Kaggle側metadataをpull検証した。
- [x] version 1の15 / 15 models完了を監視し、logsと小さい監査生成物を回収した。
- [x] pooled/fold/scope/by-well、feature/model/OOF SHAを照合し、固定gateをFAIL判定した。
- [x] same-OOF救済、inference、submissionなしで終端閉鎖した。
