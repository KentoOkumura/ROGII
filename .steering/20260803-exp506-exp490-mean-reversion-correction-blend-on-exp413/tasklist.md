# タスクリスト

## TODO

- なし。exp506 primary仮説は終端閉鎖済み。

## 進行中

- なし。

## ブロック中

- inference / submission: Stage A primary all-AND gate FAILのため実施しない。
- weight / component / scope / gate rescue: 事前契約により禁止。

## 完了

- 2026-08-03: 関連するexp490 / exp357 / exp413 familyの記録を横断確認した。
- 2026-08-03: primaryを`anchor + lambda * (exp490-exp357)`、lambdaをother-four-fold
  closed-form / `[0,0.10]`、direct convex blendをreport-onlyへ固定した。
- 2026-08-03: fold / scope / tail / lambda stabilityの全AND gateと再現性契約を固定した。
- 2026-08-04: exp497 Stage E gate FAILを確認し、exp413 Stage D保存OOFをanchorへ固定した。
- 2026-08-04: compact self-contained Stage A source、候補Notebook、契約テストを実装した。
- 2026-08-04: ユーザー承認後、正規train Notebook、canonical CPU/internet-off packageを作成した。
- 2026-08-04: version 1は最終metrics表示のNumPy bool serializationでERROR。
- 2026-08-04: 科学ロジック不変で`to_jsonable()`を通す修正と回帰テストを追加し、focused 8 tests、
  Jupytext、py_compile、Ruff、strict validatorをPASSした。
- 2026-08-04: 同じcanonical kernelのversion 2（id_no `129631767`）をCOMPLETEした。
- 2026-08-04: primary CV`7.902068462119896`、anchor比`+0.017265667715181 ft`悪化、
  nonworse`3/5 folds`、scope`0/5`、worst well`+1.816049513 ft`、deployment lambda`0.0`を確認した。
- 2026-08-04: technical checks全PASS / scientific all-AND gate FAILとして
  `FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`で終端した。
- 2026-08-04: version 2生成物を取得し、manifest SHA、OOF行数・fold・null、prediction logical SHAを監査した。
- 2026-08-04: result / metrics / SESSION_NOTES / experiment summary / backlogを終端結果へ更新した。

## 後続候補

- 必要なら、固定10% report-only convex controlのscope / tail寄与だけを説明する
  saved-artifact-only readoutをP4で別設計する。exp506 gate再評価、weight再fit、推論候補化は禁止。
