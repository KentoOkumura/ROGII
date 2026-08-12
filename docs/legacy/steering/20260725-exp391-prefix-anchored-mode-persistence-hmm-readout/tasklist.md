# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- Stage B full 773-well runはStage A1 technical / mechanism gate failureにより閉鎖。
- inference / submissionは本実験のscope外。

## 完了

- `exp391_prefix_anchored_mode_persistence_hmm_readout`を採番した。
- `docs/legacy/steering/20260725-exp391-prefix-anchored-mode-persistence-hmm-readout/`を作成した。
- experiment scaffoldを作成した。
- Routeを`pf_beam`、親をexp209 exact HMMに固定した。
- Stage A0 / A1 / B、mode identity、停止条件、実行量、禁止事項を設計に固定した。
- `docs/06_reproducibility.md`に沿う再現性設計を記録した。
- `KAGGLE_DIRECTION.md`のアイデアバックログへ登録した。
- 2026-07-25: ユーザーの`exp391を実装してください`をimplementation-only承認として記録した。
- 2026-07-25: 正規Notebookを上書きせず、11章・3,407行のJupytext percent形式
  compact self-contained train候補とfail-closed inference候補を実装した。
- 2026-07-25: Stage A0 strict resolver / join / event census / fold quota selection、
  exp209 exact joint posterior、same-pass mean / MAP / Viterbi、top-2 basin、
  transition-overlap lineage、no-switch conditional decoder、truth-late Stage B gateを実装した。
- 2026-07-25: mass-rank swap、start-prior anchor、gradual cross-mode drift、
  merge / split、tie-break、row-order、truth-read ledger、SHA、
  exp209 marginal parity、exp270 global Viterbi parityを専用14 testsで確認した。
- 2026-07-25: Ruff、F821、py_compile、Jupytext round-trip、strict experiment
  validationをPASSした。
- 2026-07-25: 正規train Notebookを採用し、Kaggle private CPU Stage A0
  version 2（id_no `128527913`）を完了した。3,783,989 rows / 773 wellsの
  strict join、全technical gate、truth/error/hidden-like事前read 0をPASSし、
  1,234 events / 730 wellsと5 foldsを覆う固定16 wellsを凍結した。
- 2026-07-25: ユーザーの`Stage A1に進んでください`を、固定16 HMM wellsの
  private CPU Stage A1実行承認として記録した。Stage B、inference、
  submissionへの承認には拡張しない。
- 2026-07-25: Stage A1を同kernel version 3（id_no `128527913`）で完了した。
  16/16 HMM wells、kernel runtime 18,105.382秒、peak RSS 4.132145 GB。
  same-pass parity、posterior normalization、projected runtime、
  HMM-supported fraction / fold countをFAILし、technical / mechanismともFAIL。
- 2026-07-25: HMM-supportedは1/19 events・1/5 folds、causeはposterior
  averaging 1 / transition 0 / K16 0 / fixed blend 3 / unresolved 15。
  78,866 candidate rowsを全行fail closedし、Stage B、inference、
  submissionなしでbranchを閉じた。
- 2026-07-25: mode ledger、decoder manifest、candidate、posterior/path ledgerの
  logical / decompressed SHAとversion 3成果物取得先をSESSION_NOTESへ記録した。
