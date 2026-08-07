# タスクリスト

## TODO

### Stage A package / run（承認済み）

- [x] 正規Notebook採用前に別承認を得る。
- [x] Kaggle package、Stage A private CPU run前に別承認を得る。
- [x] canonical packageを生成・監査する。
- [x] Kaggle private CPUへpushし、完了と固定gateを確認する。

### Stage A PASS後だけ（FAILのため実施しない）

- hidden-dynamic current-test向けのSG-only inference統合を設計する。
- current-test row order、short-well、prediction-start continuity、SciPy versionをfail-fast検証する。
- inference candidateをJupytext percent形式で別名実装・検証する。
- Kaggle inference run後にsubmit-checkを実施する。
- competition submission前に別承認を得る。

### 条件付きwell router（exp508では着手しない）

- exp508 primary all-AND PASSと独立なtarget-free complementarity evidenceを先行条件にする。
- 別exp / steeringでouter-train refit、outer-valid apply、公開固定threshold/map禁止を設計する。
- 実装・実行はさらに別承認とする。

## 進行中

- なし。

## ブロック中

- なし（Stage A package / CPU runは2026-08-04に承認済み）。
- inference / submission: Stage A primary all-AND FAILのため閉鎖。
- well router: exp508 PASSの先行条件未達のため作成しない。

## 完了

- 2026-08-04: exp413、exp497、公開source、既存well selectorの記録を横断確認した。
- 2026-08-04: selectable primaryを固定SG61/p3の1本へ限定した。
- 2026-08-04: tau85 warmup単独とwarmup+SGをreport-onlyへ固定した。
- 2026-08-04: full public 60/40 LikPF recipeとwell routingをexp508から除外した。
- 2026-08-04: 評価scope、prediction-start continuity、promotion gate、FAIL時の閉鎖方針を固定した。
- 2026-08-04: 再現性設計を`design.md`へ記録した。
- 2026-08-04: steeringと実験scaffoldを作成した。実装コードは作成していない。
- 2026-08-04: ユーザーの`exp508を実装してください`をStage A実装承認として記録した。
- 2026-08-04: 別名のcompact self-contained Jupytext source / Notebook候補を実装した。既存の正規Notebookは上書きしていない。
- 2026-08-04: exp413 OOF / fold / hidden-like resolver、SHA guard、truth-free prediction freeze、truth-late readoutを実装した。
- 2026-08-04: 固定SG61/p3、report-only tau85 / tau85+SG、pooled / fold / scope / tail / prediction-start / trajectory診断、all-AND gateを実装した。
- 2026-08-04: dedicated contract test 10件、py_compile、Ruff、Jupytext round-trip、strict experiment validation、template validationをPASSした。Kaggle runは行っていない。
- 2026-08-04: ユーザーの`実行してください`を正規Notebook採用・Kaggle package・Stage A CPU run承認として記録した。
- 2026-08-04: 55文字canonical titleはKaggle上限50文字でSaveKernel 400。旧ID未作成を確認し、意味を保持する`exp508-exp413-public-sg61p3-audit-train`へid/titleを同時短縮した。
- 2026-08-04: canonical private CPU kernel version 1（id_no `129625989`）を完了まで監視した。0 model / booster / HMM / PF / Beam / GPU、親再学習0。
- 2026-08-04: SG61/p3は保存exp413 `7.884802794→7.878669067`、gain `0.006133728 ft`、5/5 folds、固定5 scope、安全性・technical gateをPASSしたが、固定gain `>=0.01 ft`だけをFAILした。
- 2026-08-04: `FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`として、report-only救済、router、inference、submissionなしで終端閉鎖した。
