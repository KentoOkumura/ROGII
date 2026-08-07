# exp487_time_varying_gr_affine_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: Stage 0全technical gate PASS・Stage 1別承認待ち
- CV / LB / Submit: なし
- 親: exp417、実装参照・保存control: exp404
- Kaggle kernel:
  `kentookumura/exp487-time-varying-gr-affine-likelihood-pf-train` version 5

## 仮説

exp345 causal / exp350 RTSのtime-varying affine scheduleをparticle GR emissionへ
適用すると、failed exp211 static affineより局所GR driftへ追随できる。

## 変更点

- A: causal EKFの`a_t,b_t`でemission centerを変える。
- B: extended RTSでsmoothした`a_t,b_t`を使う。

両者は独立報告し、同じOOFからwinnerを選ばない。

## 検証方針

private Kaggle CPUでfixed32を実行し、2×32 = 64 PF well-runs、
8,192 seed-well trajectories、4,096,000 particle startsを完了した。
全15 technical checksはPASSし、64 variant-wellsをtruth attach前にfreezeした。
runtimeは`1,191.088 sec`、peak RSSは`1.849 GB`だった。

## 所見

fixed32記述RMSEはcausal `12.634360`、RTS `13.391424`、
saved exp404 control `9.616741`。これはCVではないが、両candidateともcontrolより
大きく悪化しており、Stage 1の性能見通しは弱い。

## 成果物

- canonical self-contained train notebook / Jupytext source
- fail-closed self-contained inference候補
- 専用contract / synthetic E2E tests
- `metrics.json` / `result.md` / `SESSION_NOTES.md`

## リスクと境界

raw GRをschedule updateとPF emissionへ二重利用する過信riskがある。
Stage 1はStage 0全PASSによりtechnical eligibilityを得たが、別承認が必要である。
inferenceとsubmissionも未承認で、すべて無効のまま保持する。
