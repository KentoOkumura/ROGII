# タスクリスト

## TODO（別承認が必要）

- なし。追加40 booster rerun、downstream、inference、submissionは未承認で
  scientific gate FAILのため計画しない。

## ブロック中

- なし。

## 完了

- backlog項目を作成した。
- 実験scaffoldを作成した。
- requirements、design、tasklistを作成した。
- 12候補semantic replacement contractを固定した。
- 実行量とcontrol非再学習を固定した。
- 再現性、leakage、scientific gateを固定した。
- scaffold時点ではcanonical notebookをmarkdown-only placeholderにし、
  実行承認までは上書きしなかった。
- 2026-07-31のユーザー依頼でcompact self-contained train候補の実装承認を得た。
- `src/exact_hmm_full_replacement.py`にfixed12 overlay、4/8 parity、
  formula/global-key/truth-late/固定88/74 schema/scientific gateを実装した。
- 別名Jupytext sourceとipynb候補を作成し、実行承認後にcanonicalへ採用した。
- 専用test 9件でcost、contract、親Stage C runtime parity、SHA allowlist、
  4/8 parity、missing key、
  Stage A固定schema、scientific gate、canonical非上書きを検証した。
- 2026-07-31のユーザー依頼でcanonical notebook採用、Kaggle package/push/runの
  承認を得て、1 variant / 2 objectives / outer 5 x inner 4 = 40 CPU booster、
  control再学習0を再確認した。
- Kaggle private CPU version 1（id_no `129217774`）を実行し、Stage C、
  technical/leakage/score guard、科学readoutを完了した。
- hard primary `8.652531956 -> 8.639368546`、改善3/5 folds、
  by-well p95 `+0.381470357 ft`、worst `+4.254514134 ft`でscientific gate FAIL。
- post-readout feature importance列名バグをcanonical sourceで修正した。
- branchをcloseし、backlogと実験記録を更新した。
