# exp487_time_varying_gr_affine_likelihood_pf 結果

## 仮説

time-varying affine emission centerがPFの局所GR drift耐性を改善する。

## 設定

- Variant A: exp345 causal EKF schedule
- Variant B: exp350 bidirectional extended RTS schedule
- PF: 500 particles、128 seeds、temperature 5、sigma x1.0
- Stage 0: fixed32、2 variants × 32 wells = 64 PF well-runs
- Kaggle: private CPU、8 workers、GPU / internet off

## Stage 0結果

Kaggle kernel version 5（id_no `129180524`）は`COMPLETE`となり、Stage 0の
technical gateは全PASSした。

| variant | fixed32 RMSE | controlとの差 |
| --- | ---: | ---: |
| causal EKF affine emission | 12.634360 | -3.017619 ft |
| bidirectional RTS affine emission | 13.391424 | -3.774684 ft |
| saved exp404 control | 9.616741 | - |

fixed32は156,088行・32 wellsの記述評価であり、CVではない。両candidateとも
saved exp404 controlより大きく悪化したため、technical eligibilityは満たすものの、
Stage 1の性能見通しは弱い。

実行量は64 candidate PF well-runs、8,192 seed-well trajectories、
4,096,000 particle starts。control PF、HMM、Beam、model、booster、GPUの再実行は
すべて0だった。実測runtimeは`1,191.088 sec`、peak RSSは`1.849 GB`、
全773 wellsへの投影は`28,772.209 sec`だった。

## Technical gate

全15 checksがPASSした。主な確認結果は次の通り。

- prediction / schedule / PF ledgerは64 variant-wellsすべてtruth attach前にfreeze。
- freeze前のtruth、error、outcome fold、hidden role読込はすべて0。
- causal schedule、RTS forward parity / terminal / covariance、stable seedを確認。
- fallback wellsは0、causal / RTSのscale clip最大率はともに0。
- causal boundary jump sigma p95は`0.004974`。
- finite coverage、SHA、runtime projection、RSSを確認。

## 再現性

- scientific contract SHA:
  `18743aff469f4ca1a410fdc3dda62261faccdbda020410185f43f062f83a79e3`
- input manifest SHA:
  `82a39670ba6c69d944f6f4832499b211bacda2f6f14ccfca014816b709cff743`
- freeze manifest SHA:
  `1366aca8282bd003e56df255e96e3c87a499fab57838643581e7744051597aa2`
- process-noise logical SHA:
  `aae37e5eecfc220d6c345b96000ddba395bc7eb2ecd2a8b1011f8fae1e16bca8`
- stable seed:
  `sha256_first16("likpf::train::<well_id>")`、variant名を除外
- deterministic anchor: no

## 解釈

dynamic affine schedule自体、truth-late境界、PF emissionへの適用は技術的に成立した。
一方、fixed32記述値ではcausalもRTSもcontrolを明確に下回った。特にRTSはcausalより
さらに`0.757065 ft`悪く、future raw GRを使うsmoothingがこのPF emissionで有利という
証拠は得られていない。raw GRをschedule updateとPF likelihoodに二重利用する
過信riskと整合的な結果である。

## 判断

Stage 0契約上はStage 1 eligibleだが、自動昇格はしない。Stage 1、inference、
submissionは未承認・無効のまま保持する。Stage 1へ進む場合も2 variantを独立報告し、
same-OOF winner selectionやparameter / sigma / temperature救済は行わない。
