# exp403 タスクリスト

## 実装完了

- [x] 別名Jupytext percent形式のcompact self-contained train候補を作る。
- [x] exp263 / exp333 / exp355 input resolverとSHA guardを実装する。
- [x] exp263 generation fold partition単位のstreaming joinを実装する。
- [x] reporting fold / generation foldの独立ledger監査とcross-tabを実装する。
- [x] truth前freeze、exp263 parity、full replacement、correction content SHAを実装する。
- [x] 固定9 λ、outer-train eligibility、最大eligible λ、zero fallbackを実装する。
- [x] fold/scope/hidden-like/by-well/persistent episode/recovery gateを実装する。
- [x] fail-closed inference候補と専用contract testを作る。
- [x] Jupytext、py_compile、Ruff、専用testを実行する。
- [x] strict validate-expを実行する。

## 実行完了

- [x] ユーザーのKaggle train実行承認と正確な実行量を記録する。
- [x] 正規train Notebook採用を承認範囲に含める。
- [x] 正規train Notebookを採用し、再検証する。
- [x] packageのmetadata / input source / bootstrap configを検証する。
- [x] Kaggleへpushし、version 4のCOMPLETEまで監視する。
- [x] scientific gateとSHAを記録する。

## Terminal close

- scientific promotion: FAIL。
- positive λ: 0 / 5 folds、cross-fit λは全fold 0。
- inference: 実施しない。
- submission: 実施しない。
- 同一OOFのλ / weight / gate / router救済: 禁止。

## 完了

- `kaggle-review-exp`と`kaggle-strategy`の手順を確認した。
- `docs/agent-playbooks.md`と`docs/06_reproducibility.md`を確認した。
- exp403番号、`ensemble` route、親とbranch lineageを確定した。
- 保存済みOOFの自然置換diagnosticを根拠として記録した。
- λ候補、cross-fit、tie-break、fallback、promotion gateを確定した。
- fold ledger独立性、truth-late、SHA、実行量、禁止事項を確定した。
- steering、experiment scaffold、backlog、summaryを作成した。

## Kaggle train実行承認に含めない

- settings.pyの編集
- model / PF / HMM / Beamの新規実行
- inference、submission生成、competition submit
