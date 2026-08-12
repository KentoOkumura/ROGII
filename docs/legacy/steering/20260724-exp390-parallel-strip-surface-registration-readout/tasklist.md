# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- 773-well full run、Stage 1 / 2、inference、submission:
  Stage 0 sparse two-sided support gate FAILにより停止。
- angle/distance/overlap/min donor/bandwidth/Huber/one-sided/blend/selector救済:
  事前禁止事項のため実施しない。

## 完了

- 2026-07-24: 次の空き番号`exp390`を確認した。
- 2026-07-24: `docs/legacy/steering/20260724-exp390-parallel-strip-surface-registration-readout/`を作成した。
- 2026-07-24: `experiments/exp390_parallel_strip_surface_registration_readout/`を作成した。
- 2026-07-24: route、parent、parallel-strip座標、pair条件、単一candidate、
  Stage 0/1/2、再現性、停止条件、禁止事項をdesign-onlyで固定した。
- 2026-07-24: `docs/06_reproducibility.md`を確認し、RNGなし・stable key・SHA契約を設計へ反映した。
- 2026-07-24: ユーザーの`exp390を実装してください`をimplementation-only承認として記録した。
- 2026-07-24: 正規Notebookを上書きせず、10章・Jupytext percent形式・
  2,270行のcompact self-contained train候補とfail-closed inference候補を実装した。
- 2026-07-24: PCA axis canonicalization、modulo-π angle、overlap、same-s補間、
  two-sided Huber fit、prefix-only校正、exp226 exact fallback、truth-late境界を
  専用contract test 10件で確認した。
- 2026-07-24: compact train/inferenceのJupytext round-trip、`py_compile`、
  Ruff、専用pytest、strict experiment validationをPASSした。
- 2026-07-24: ユーザーの`実行してください`を正規train Notebook採用、
  private CPU / internet offの16-well Stage 0 preflight承認として記録した。
- 2026-07-24: 1 candidate / 5 reporting folds / 16 preflight wells /
  model・HMM・PF・Beam・booster・parent replay各0を確認した。
- 2026-07-24: canonical metadata、kernel sources、bootstrap ZIP、embedded config/source、
  canonical Notebook本文をpush前に監査した。
- 2026-07-24: Kaggle version 1は3件のtest directoryを選ぶinput resolver不具合で
  scientific処理前に停止した。773件train directoryを件数で一意選択する
  fail-closed修正と回帰testを追加し、専用test `11 passed`を確認した。
- 2026-07-24: Kaggle version 2（id_no `128480051`）は`COMPLETE`。
  16 wells / 73,586 rowsを`60.401419 sec`で処理し、leakage/read、
  fallback、angle、overlap、runtime、RSS gateをPASSした。
- 2026-07-24: eligible pairは8/16 queries・合計10・query最大2で、
  4 donorかつ正負両側supportを満たすnodeは0。two-sided row/well coverageと
  donor p05の3 gateをFAILし、`stage0_failed_closed_sparse_two_sided_support`で終了した。
- 2026-07-24: 必要な小さいKaggle outputだけを取得し、input、fold、geometry、
  pair、node、fit、calibration、role-read、guard、summary、manifest SHAを記録した。
- 2026-07-24: 完了済みexp390を`KAGGLE_DIRECTION.md`の実行待ちbacklogから削除し、
  threshold救済をしない0-fit全well support censusを低優先の別アイデアとして追加した。
