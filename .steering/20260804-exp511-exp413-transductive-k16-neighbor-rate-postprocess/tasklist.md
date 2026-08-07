# タスクリスト

## 目的

固定済みのpredicted-only K16 neighbor-rate後処理を、承認境界を越えず実装・評価する。

## TODO

- なし。性能gate FAILにより終端閉鎖済み。

## 進行中

- なし。

## ブロック中

- なし。inference / submissionは未承認ではなく、固定fail-close判断により対象外。

## 完了

- requirements / designへ数式、parameter、validation、gate、再現性、リスクを固定した。
- Jupytext percent形式のcompact self-contained train実装と正規train Notebookを作成した。
- exp413 OOF/fold/raw geometryのSHA-qualified resolverと`X/Y/Z` strict allowlistを実装した。
- K16 field、self exclusion、stable local-linear solve、support ledger、固定alpha/capを実装した。
- prediction freeze前のtruth/role/error access 0とtruth-late接続を実装した。
- pooled/fold/MD/hidden-like/by-well/support/continuityと固定all-AND gateを実装した。
- contract test 12件、py_compile、Ruff、Jupytext round-trip、strict validatorをPASSした。
- Kaggle version 1のcompetition mount、version 2のweight underflow、version 3のexp115 source
  欠落を、科学parameterを変えず技術修正した。
- private CPU version 4をCOMPLETEし、artifact、metric、content SHAを取得・検証した。
- technical gateは全PASS、CV `7.883964795205812`、pooled gain `0.000837999 ft`、
  nonworse folds `2/5`として性能gateをFAILした。
- same-OOF rescue、inference、submissionを行わず終端閉鎖した。
