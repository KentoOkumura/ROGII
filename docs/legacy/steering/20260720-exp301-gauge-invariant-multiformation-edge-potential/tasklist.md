# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。Stage 0 FAILによりbranchを閉じ、inference、submission、案2/案3の開始条件を満たさない。

## 完了

- [x] exp301採番と既存experiment/steeringとの衝突なしを確認した。
- [x] 仮説、数式、fold-safe境界、固定数値contract、success/failure policyを設計した。
- [x] 案2/案3を案1配下の`reserved_followup_contract.md`へ固定する方針を確定した。
- [x] 再現性設計を`design.md`へ記載した。
- [x] 2026-07-20のユーザー依頼によりexp301実装を承認済みにした。
- [x] Stage 0 identity/support auditのJupytext percent形式self-contained train sourceを実装した。
- [x] outer-valid/test safe loaderを実装し、formation 6列、GR、true TVTのpre-freeze accessをfail-closedにした。
- [x] stride 16 multiformation edge builder、250 ft active grid、bilinear basis、active/query component donor auditを実装した。
- [x] fixed inner 3-way lambda selectionとdeterministic Huber-IRLS sparse solverを実装した。
- [x] exp226 fold/parity比較、hidden-like、by-well、exp293 H512 add-one oracle diagnosticを実装した。
- [x] gauge shift、affine recovery、formation permutation、valid poison、fold/same-name exclusion、no-donor component、stable SHAのunit testsを追加した。
- [x] compact self-contained notebook候補を別名で生成し、正規placeholderを維持した。
- [x] Jupytext test、py_compile、ruff、Notebook tests、strict `make validate-exp`をPASSした。
- [x] active variant 1、outer folds 5、inner solver候補3、LightGBM config 0、booster 0、parent/control再学習0を再確認した。
- [x] ユーザー承認済みcompact候補を正規train notebookへ採用した。
- [x] Kaggle package bootstrap 16ファイルとloose/package/config SHA一致を確認した。
- [x] exp226/exp263/exp115親kernel metadataをKaggleから取得し、接続先を確認した。
- [x] full-name slugの`SaveKernel 400`では実行未作成を確認し、既知の長slug制約として短縮slugへ切り替えた。
- [x] version 1の`KeyError: 'x_start'`をfiltered edge DataFrameの1行型バグへ特定し、局所修正と回帰テストをPASSした。
- [x] 同一canonical kernel version 2を実行し、Kaggle status `COMPLETE`とid_no `128007163`を確認した。
- [x] Stage 0 identity/leakage/runtime guards PASS、component donor coverage 2 guard FAILを確認した。
- [x] Stage 0 outputを取得し、input/identity/supportのlogical SHAとfile SHAを記録した。Stage 1未実行のためgrid solution/solver/OOF/gzip SHAは対象外と記録した。
- [x] fail-closed policyどおりsolver fit 0でStage 1を開始せず、inference、submission、案2/案3を閉じた。
