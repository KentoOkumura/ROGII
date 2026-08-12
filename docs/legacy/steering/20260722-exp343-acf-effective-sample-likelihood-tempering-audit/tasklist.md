# タスクリスト

## 未着手

- なし。同一契約の再実行や救済gridは追加しない。

## ブロック中

- Stage 1、inference、submissionはStage 0固定gate FAILにより不適格。

## 完了

- 2026-07-22: exp343として採番し、steeringと実験scaffoldを作成した。
- 2026-07-22: lag、fallback、shrinkage、clip、Stage gate、救済禁止を設計固定した。
- 2026-07-23: Stage 0 compact self-contained train、fail-closed inference、
  正規Notebook、専用contract testを実装した。
- 2026-07-23: pairwise Pearson、raw-row last-512、contiguous finite run、
  outer-train fold median、joint-evaluable stability、worst-window gateをコードで固定した。
- 2026-07-23: ユーザー承認に基づきKaggle private CPU version 1を実行した。
- 2026-07-23: 773 wells中joint-evaluable 295、fallback 478、stable fold 0/5、
  upper-clip率full 0.997413 / tail 1.0を記録した。
- 2026-07-23: 固定gate FAIL、`stage_0_failed_close_without_rescue`としてbranchを閉じた。
- 2026-07-23: HMM、model、booster、control再実行、Stage 1、推論、提出を0のまま維持した。
- 2026-07-23: 旧exp320をreopenせず、Type Well群非依存のnegative resultとして記録した。
