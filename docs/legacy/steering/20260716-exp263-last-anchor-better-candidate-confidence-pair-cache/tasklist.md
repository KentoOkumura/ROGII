# タスクリスト

## TODO

- exp264 Stage Bは別承認とし、1 variant × 2 objectives × 5 folds = 10 CPU boostersを再提示する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- 実装契約、candidate blend audit、再現性 guard を確認した。
- 再現性設計を `design.md` に記入した。
- 学習0 variant / 0 config / 0 fold training / 0 booster / control 再学習0を確認した。
- core 12 source/value/confidence resolver、candidate-major writer、outer-train-only eligibilityを実装した。
- 8 pair、w500 alias、3 named formula DAG、chunked virtual loader、selectable guardを実装した。
- Jupytext percent形式のStage 0/1 sourceから正規train/inference notebookを生成した。
- synthetic end-to-endとParquet hash round-tripを含むexp263 tests、repo全tests、ruff、py_compile、Jupytext testをPASSした。
- `validate-exp` strictとtrain/inference canonical package生成をPASSした。
- bootstrap 10 filesにconfig、builder、contract、loader、schema、loader docsが入ることを確認した。
- 12 Kaggle kernel sourceを固定し、private CPU version 1を完走した。
- full 3,783,989 rows / 773 wells、source 9 groups / 12 gzip、value/confidence各60 partitionsを確認した。
- source decompressed SHA、partition file/content/schema SHA、catalog/manifest SHAを記録した。
- 代表4 Parquetとvirtual best-pair loaderを実ファイルで再検証した。
- Stage 1 inference v2で6 primitive / 5 pair / fixed formulaとsubmissionを完走し、値parityを確認した。
- submission ref `54761954`のscoring完了を確認し、Public LB 7.800を記録した。
- Stage 1 inference v3で21 namespaced confidence列を実値検証し、v2旧15列exact parityとsubmission byte parityを確認した。
- exp264 current-test confidence前提を満たし、exp263 Stage 1を完了した。v3のcompetition submitは行っていない。
