# 要件

## 依頼

rate追従遅れを減らす第2案として、exp209の局所rate kernelへ固定低確率の
対称broad jumpを混ぜる`exp442_symmetric_broad_jump_rate_transition_hmm`の
compact self-contained実装候補とcontract testを作る。

2026-07-29のユーザー依頼「exp442を実装してください」により実装候補を作成した。
2026-07-30のユーザー依頼「steeringと検証契約を更新してから実行してください」により、
exp442をexp441の救済ではない独立仮説へ再定義し、正規train Notebook採用、
Kaggle private CPU package、fixed32 Stage 0実行までを承認済みとする。
Stage 1、inference、submissionは別承認とする。

## 仮説

大部分を親transitionに残し、稀な全support branchだけを加えると、
通常区間の安定性と急変時の到達性を両立できる。

exp441の全support OU置換とは異なり、exp442はexp209の局所kernelを99%維持し、
独立した1% escape branchを追加するdefensive mixture仮説である。exp441のFAILは
negative contextとして保持するが、exp442の実行前提にはしない。

## 制約

- Route `pf_beam`、親/controlはexp209。
- scientific candidateは`jump_weight=0.01`、`sigma=0.02`の1本だけ。
- broad branchはparent Euler conditional meanを中心とする対称Gaussian。
- GR innovation方向triggerを使わない。
- parent local kernel、TVT transition、state、grid、emission、prior、readout固定。
- weight/sigma grid、reset、re-anchor、datum branch、selector/blend禁止。
- exp441の結果をpositive evidenceやgate救済に使わない。
- 正規train Notebook採用、Kaggle private CPU package、fixed32 Stage 0は承認済み。
- Stage 1、inference、submissionは未承認。

## 受け入れ基準

- mixture式、weight、sigma、境界mass、branch responsibilityが一意である。
- fixed32 32 candidate runs、parent rerun 0を固定する。
- direction、episode SSE、control safetyのAND gateとfail actionが固定されている。
- truth-late、SHA、実行量、禁止事項が全文書で一致する。
- compact self-contained trainを正規train Notebookへ採用し、Kaggle package内の
  config、CPU、internet無効、kernel source、入力SHA契約を検証する。
- 現在configではStage 0だけを実行可能とし、Stage 1、inference、submissionは
  fail-closedのままにする。
