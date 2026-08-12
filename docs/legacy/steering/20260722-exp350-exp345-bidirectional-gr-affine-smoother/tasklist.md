# タスクリスト

## 完了

- 2026-07-22: full-well GR bidirectional smootherを、閉鎖済みexp345の再開ではなく独立`exp350`として採番した。
- 2026-07-22: steering、実験scaffold、offline input contract、単一変更、extended RTS式、truth-free freeze、control再利用、実行量、AND gate、FAIL後no-rescueを固定した。
- 2026-07-22: `KAGGLE_DIRECTION.md`へ低・P3 design-only候補として追加し、`experiment_summary.md`へ反映した。
- 2026-07-23: ユーザーの実行承認により、exp345 compact self-contained sourceを構成参照元としてStage 0 trainを実装した。
- 2026-07-23: exp345 artifact SHA preflight、forward state/covariance再生成、saved schedule parity、fixed-interval RTS、numerical audit、candidate HMM 1回、late truth/role join、promotion gateを実装した。
- 2026-07-23: compact候補を正規train Notebookへ採用した。親と同じ11章構成を維持し、同一exp helper importと`__file__`依存を持ち込んでいない。
- 2026-07-23: py_compile、ruff、4件の専用test、Jupytext round-trip、strict experiment validation、Kaggle package metadata監査をPASSした。
- 2026-07-23: 1 variant / forward 773 / smoother 773 / new HMM 773、control HMM再実行0、LightGBM/fold/booster/PF/Beam/GPU各0を記録してKaggle CPU version 1を実行した。
- 2026-07-23: canonical kernel `kentookumura/exp350-bidirectional-gr-affine-smoother-train` version 1、id_no `128274195`の`COMPLETE`を確認した。
- 2026-07-23: technical gate PASS、scientific gate FAIL、decision `stage_0_failed_close_without_rescue`を確認した。
- 2026-07-23: result、metrics、SESSION_NOTES、README、config、KAGGLE_DIRECTION、experiment_summaryへ結果を反映し、完了済みbacklogを削除した。

## 判定

- candidate RMSE: `14.367548324`。
- masked parent比: `+0.133499460 ft`、5/5 folds、hidden-like 2面改善でPASS。
- exp345 causal比: `-0.036005627 ft`、2/5 foldsでFAIL。
- parent比by-well median: `-0.008671745 ft`でPASS。
- parent比by-well p95: `+1.346426592 ft`でFAIL。
- worst `8995c945`: parent比`+20.887374 ft`でFAIL。
- technical gate: SHA、parity、coverage、finite、terminal、covariance、runtimeを全PASS。
- scientific gate: FAIL。

## 閉鎖後の禁止事項

- Stage 1、inference、submission。
- version 2、exp345のreopen/reparent/re-push、control HMM再実行。
- Q / rcond / clip / covariance floor / smoother回数のgrid。
- causal/bidirectional blend、row/well gate、fallback selector、post-hoc rescue。
- true TVT/error/oracleを使うstate、smoother、clip、gate、停止条件。

## 次のアクション

branchを閉じる。同じaffine-smoother familyの救済候補は追加せず、独立した既存P1--P2候補exp340を優先する。

