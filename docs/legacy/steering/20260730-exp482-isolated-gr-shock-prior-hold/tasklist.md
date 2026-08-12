# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage A0 eligibility FAILのため、Stage A1、Stage 1、inference、
  submission、threshold/window/control定義の救済、再runはterminal block。

## 完了

- `exp482_isolated_gr_shock_prior_hold`を採番した。
- exp440と独立したraw-GR単発shock仮説として位置付けた。
- raw-shock、past/future agreement、current-emission conflictのAND triggerを固定した。
- trigger rowだけleave-one-current-observation-out meanへ置換し、
  親HMM stateと後続予測を変更しない介入を固定した。
- raw-only Stage A0、target-free fixed64、Stage A1、Stage 1の段階を固定した。
- Stage A0/A1/1のvariant、HMM replay、control rerun、model/booster/PF/Beam/GPU数、
  technical/scientific gate、no-rescue actionを固定した。
- `docs/06_reproducibility.md`を確認し、RNGなし、truth-late freeze、
  logical/decompressed SHA、初回run非anchorを記録した。
- steering 3文書を作成した。
- design-only experiment scaffoldを作成した。
- `config.yaml`、README、SESSION_NOTES、result、metricsをdesign-onlyへ更新した。
- `KAGGLE_DIRECTION.md`の未着手backlogへ低・P3として追加した。
- `experiment_summary.md`へdesign-only実験を反映した。
- compact self-contained Jupytext train候補を別名で実装した。
- fail-closed inference guard候補を別名で実装した。
- raw-only census、fixed64 manifest、unchanged exp209 message replay、
  leave-one-out readout、truth-late gateの専用testを作成した。
- Jupytext round-trip、`py_compile`、Ruff F821/E9、専用pytest`14 passed`、
  strict experiment validationを完了した。
- 正規Notebook scaffoldを上書きせず、package/runを行っていない。
- ユーザーの「実行してください」によりcanonical train Notebook採用、
  Kaggle package、Stage A0/A1 private CPU runを承認済みに変更した。
- compact self-contained sourceからcanonical train Notebookを生成した。
- strict train packageを作成し、metadata、埋め込みconfig、実行量契約を検証した。
- canonical private CPU kernel v1へpushし、Kaggle `id_no=129168015`を確認した。
- kernel v1を`COMPLETE`まで監視した。
- raw-only census 773 wells、isolated shock 17,047 rows、support 763 wells、
  zero-shock control 10 wellsを記録した。
- zero-shock control最小32 wells gateをFAILし、HMM replay 0のまま
  `stage_a0_eligibility_failed_closed`で停止した。
- raw census / raw-shock rows SHAと実行量を記録し、exp482をterminal closeした。
