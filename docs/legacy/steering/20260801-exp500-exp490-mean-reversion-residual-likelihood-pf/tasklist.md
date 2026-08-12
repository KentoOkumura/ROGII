# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp500_exp490_mean_reversion_residual_likelihood_pf`として採番した。
- `pf_beam` route、科学的PF親exp486、平均回帰機構親exp490を固定した。
- 1 scientific variantの状態式、初期化、更新順序、固定PF契約を確定した。
- Stage 0 fixed44、Stage 1 full 773-wellの実行量、gate、fail-close、禁止救済を確定した。
- `docs/06_reproducibility.md`に従うstable seed、固定順序、SHA、probe rerun方針を確定した。
- steering、実験scaffold、`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`をdesign-only状態で作成した。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へ設計を反映した。
- ユーザーの実装承認後、exp486 compact self-contained trainを構成参照として、
  Stage 0専用Jupytext percent形式compact self-contained train候補を別名で実装した。
- `rho_t`、K16 destination境界、zero-state identity、`rho=1` exp486 float32 parity、
  stable RNG、truth/control/role-before-freezeを検証する専用contract test 6件を作成しPASSした。
- candidate Notebookを23 cells / output 0で生成し、Jupytext test、構文、ruffをPASSした。
- ユーザー承認後、candidateを正規train Notebookへ採用し、Kaggle packageを作成した。
- Kaggle private CPU version 1の展開済みexp486 CSV path technical errorを、入力内容と科学契約を
  変えないplain/gzip reader修正だけで解消し、version 2を完走した。
- Stage 0 fixed44はtechnical 13/13 PASS、persistent subset 13/16 wells・5/5 folds改善だったが、
  matched-control pooled / by-well p95とPF sentinel worst-wellの3 safety gateをFAILした。
- 事前登録どおり`stage0_fail_closed`とし、Stage 1、inference、submission、same-fixed44 rescueを
  禁止したまま終端閉鎖した。
- 2026-08-02、ユーザーがStage 0 fail-closeを理解したうえでStage 1の実装・実行を明示overrideした。
  Stage 0判定を変更せず、inference / submissionを含めない限定例外として記録した。
- Stage 1用Jupytext source、4-shard target-free freeze、strict merge、full OOF gateを実装した。
- dedicated contract 9件、py_compile、ruff F821、Jupytext、strict experiment validationをPASSした。
- 1 variant / 773 PF wells / 98,944 seed-well / 49,472,000 particle starts、control再実行0を
  `SESSION_NOTES.md`へpush前記録した。
- Kaggle private CPUの4 shardsを各version 1で完走し、773 wells / 3,783,989 rowsをfreezeした。
- merge version 1のCSV payload SHA readbackとversion 2のexp226参照列の技術不整合を、
  PFや科学契約を変えず修正し、version 3を完走した。
- full OOF RMSE `8.813504627`、exp404比`2.101017446 ft`改善、5/5 folds・全固定scope改善を確認した。
- technical 18/18 PASSだがby-well p95 `+6.653601019 ft`、worst well
  `+46.154671032 ft`でscientific tail gateをFAILした。
- `stage1_fail_closed_under_override`としてsame-OOF rescue、inference、submissionなしで終端閉鎖し、
  config、metrics、result、README、SESSION_NOTES、summary、directionを更新した。
