# exp483_huber_gr_filtering_likelihood_pf セッションノート

## 目的

exp389 Huber GR emissionを現行temperature-5 likelihood-PFのfiltering尤度へ
一因子だけ移植し、stable-hash fixed32 Stage 0 technical preflightを実行できる
compact self-contained候補を実装する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage1_gate_failed_terminal_close`
- Priority: P1
- implementation / 専用test: 実装済み
- 正規train Notebook採用 / Kaggle package / push / fixed32 Stage 0:
  2026-07-30の`実行してください`で承認され、kernel version 1で完了
- Stage 1: kernel version 2で完了、technical PASS / scientific FAIL
- inference / submission: 未承認

## 根拠と差分

- exp389: direct `11.938287235 -> 11.852741130`、5/5 folds。
- exp389 tail: p95 `+0.002234351 ft`、worst `+1.750248203 ft`でFAIL。
- exp430は凍結軌道evidenceだけをHuber化しておりfiltering尤度ではない。
- exp483は粒子ごとのGR log likelihoodだけをfixed Huberへ置換する。

## 実行契約

- Stage 0: candidate 32 PF well-runs、4,096 seed-well、2,048,000 particle starts。
- Stage 1上限: candidate 773 PF well-runs、98,944 seed-well、
  49,472,000 particle starts。
- exp404 control rerun、HMM、Beam、model、booster、GPUは0。
- 実装・実行したcandidateは1 variant。Stage 0実績は上記契約と一致した。

## 実装

- `exp483_huber_gr_filtering_likelihood_pf_compact_selfcontained_train.py`
  をJupytext percent形式で実装し、compact `.ipynb`へ変換した。
- exp404 x1.0の入力準備、particle/rate dynamics、resampling、roughening、
  T=5集約を固定し、filtering中のper-particle GR scoreだけを
  fixed Huber `delta=1.345`へ置換した。
- Huber weight更新は`log(weight)-rho(z)`をrow内maxで安定化する。
  Huber loss自体にclipやGaussian mixtureは入れていない。
- fixed32 manifestはprediction前に`well`列だけを読み、suffix truth、
  保存exp404 control、fold、hidden-like roleはfreeze前に読めない
  fail-closed ledgerを実装した。Stage 0ではfold/hidden-like roleを読まない。
- candidate prediction、well audit、scientific/input/runtime manifest、
  logical/decompressed SHAとreadbackをtruth attachment前にfreezeする。
- truth-late fixed32 RMSEはreport-onlyであり、CVやpromotion判定には使わない。
- 実装時点では既存正規`_train.ipynb`をplaceholderのまま維持し、
  別名compact Notebookを作成した。

## 検証ログ

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp483_huber_gr_filtering_likelihood_pf/\
  exp483_huber_gr_filtering_likelihood_pf_compact_selfcontained_train.py
.venv/bin/python -m py_compile <compact train.py> <dedicated test.py>
.venv/bin/ruff check <compact train.py> <dedicated test.py> --select F821
.venv/bin/pytest -q tests/test_exp483_huber_gr_filtering_likelihood_pf.py
```

- 専用test: `12 passed`。
- formulaはHuber内側でGaussianとexact一致、外側linear、large-z clipなし。
- synthetic constant-GR no-op PFはprediction / seed log-likelihood /
  resampling / ESS / position clipがGaussian referenceとbitwise一致した。
- embedded Gaussian referenceはexp404 kernelの全5配列と完全一致した。
- extreme GRでもHuber prediction / seed log-likelihoodはfiniteだった。
- fixed32 manifest SHA、stable seed、truth-late fail-close、
  prediction/audit SHA readbackをPASSした。
- local notebook実行は行っていない。Kaggle package、push、Stage 0は完了した。

## 親compactとの構成比較

- exp404 compact train: 11章、2,174行。
- exp483 compact train: 11章、1,747行。
- exp483はfull-OOF paired scale ablation/resume処理をStage 0 fixed32へ縮めた一方、
  input preparation、PF kernel、prediction freeze、truth-late、technical gate、
  artifact orchestrationの役割を維持した。
- `Path(__file__)`、同一exp helper import、薄い`main()` entrypointはない。

## 再現性

exp404 stable per-well seed、T=5、固定順、truth-late、decompressed content SHAを
継承する。Huber variant名はbase seedへ入れない。初回runはanchorにせず、
独立rerunのprediction SHA一致後だけanchor化できる。

## 次のアクション

1. Stage 0/1のtechnical gate、CV、scope、tail、SHAを記録済み。
2. 事前登録どおりsame-OOF rescueなしでbranchを閉じる。
3. inference、submissionは実行しない。

## 2026-07-30 Stage 0実行承認

- 追加依頼`実行してください`を、正規train Notebook採用、Kaggle package/push、
  fixed32 Stage 0実行の承認として記録した。
- push前の実行量:
  - active scientific variant: 1
  - candidate PF well-runs: 32
  - seed-well trajectories: 4,096
  - particle starts: 2,048,000
  - 保存exp404 control PF rerun: 0
  - LightGBM config / trained fold / booster: 0 / 0 / 0
  - HMM / Beam / GPU: 0 / 0 / 0
- `run_stage_1=false`、`run_inference=false`、`create_submission=false`を維持する。
- planned canonical kernel:
  `kentookumura/exp483-huber-gr-filtering-likelihood-pf-train`
- 同slugのpush前`kaggle kernels pull -m`は403で、既存kernelを確認できなかった。
- compact trainと正規train NotebookのSHAは
  `22aedbca8baec8a7f43f42defc82b1b0721b56623095e0cb14284f071a94d19b`
  で一致した。
- `--notebook train --run-on-push --strict --no-src`でcanonical packageを作成した。
  metadataはprivate / CPU / GPUなし / internetなし、competition inputと
  `kentookumura/exp404-v1-frozen-predictions`だけをattachした。
- bootstrap内configで`run_stage_0=true`、`run_stage_1=false`、
  `run_inference=false`、`create_submission=false`、fixed32 manifest同梱を確認した。
- 2026-07-30 12:22 UTCにcanonical kernel version 1をpushした。
  - kernel: `kentookumura/exp483-huber-gr-filtering-likelihood-pf-train`
  - id_no: `129169339`
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp483-huber-gr-filtering-likelihood-pf-train`
  - push直後の`pull -m`で存在、private、CPU、internet off、入力metadataを確認した。

## 2026-07-30 Stage 0完了

- canonical kernel version 1は2026-07-30 12:27:50 UTCに`COMPLETE`となった。
- scope: `stage0_fixed32_technical_preflight_not_cv`
- status: `stage0_technical_pass_no_automatic_stage1`
- 実行実績:
  - rows / wells: `156,088 / 32`
  - candidate PF well-runs: `32`
  - seed-well trajectories: `4,096`
  - particle starts: `2,048,000`
  - control PF / HMM / Beam / LightGBM / booster / GPU: すべて`0`
- technical gateは`10 / 10 PASS`:
  - formula unit contract
  - Huber / Gaussian inside-delta equality
  - no-op toy PF bitwise parity
  - finite prediction coverage
  - stable seed identity
  - truth / error / fold / hidden-like roleのfreeze前read 0
  - execution count match
  - artifact SHA readback
  - full-runtime projection
  - peak RSS
- fixed32 truth-late report-only:
  - candidate RMSE: `9.811671589777898 ft`
  - 保存exp404 control RMSE: `9.616740808061033 ft`
  - candidate - control: `+0.19493078171686484 ft`
  - improved wells: `18 / 32`
- 上記fixed32値はCVでもpromotion判定でもない。参考診断はnegativeだが、
  Stage 0の技術契約は満たした。
- runtime:
  - fixed32: `283.78094363212585 sec`
  - full 773 projection: `6855.08341961354 sec`
  - peak RSS: `0.4953575134277344 GB`
- SHA:
  - scientific contract:
    `089765cb14c395c1ff678d93c4a4940481aa7a8b846811287a7168ddda25d3c6`
  - prediction logical:
    `417397292feb6cbe448b196347a0902dd07a13e3a122376623da671acc5e0371`
  - prediction raw gzip:
    `e0104875db6b60e4f7b897fc4b6930e1777e200f18910df960e91d5cb481cc3d`
  - prediction decompressed:
    `5fcb88ac86c06e1ff86bd2940151bf7b1b13c8311d8cc102162ea4cda113005a`
  - well audit:
    `5a9374511c7c39eada248f709b8399cafcbbdd10023dad2fc9ed83124caad692`
- train-side確認に必要なgate、metrics、runtime、SHAはKaggle logsに揃っていたため、
  output archiveは取得していない。
- Stage 1、inference、submissionは無効のままで、実行していない。

## 2026-07-30 Stage 1実行承認

- ユーザー依頼`Stage1に進んでください`を、同じ科学契約の全773 wells
  Stage 1実装、canonical package、push/runの別承認として記録した。
- push前の実行量:
  - active scientific variant: `1`
  - candidate PF well-runs: `773`
  - seed-well trajectories: `98,944`
  - particle starts: `49,472,000`
  - reporting folds: `5`
  - 保存exp404 control PF rerun: `0`
  - 保存exp209 HMM rerun: `0`
  - LightGBM config / trained fold / booster: `0 / 0 / 0`
  - HMM / Beam / GPU runs: `0 / 0 / 0`
- Stage 1はHuber `delta=1.345`、x1.0 GR scale、500 particles、
  128 stable seeds、PF dynamics、resampling、roughening、T=5集約を変更しない。
- candidate predictionとSHAを全773 wellsでfreezeした後だけ、suffix truth、
  保存exp404 control、exp226 reporting fold、exp115 hidden-like role、
  保存exp209 HMMを読む。
- 評価はpooled RMSE、5 folds、raw-GR observed/missing、高missing wells、
  MD-since 1000+、hidden-like 2面、by-well p95/worst、fixed HMM/PF 50:50の
  事前固定AND gateだけとする。
- `run_stage_0=false`、`run_stage_1=true`、
  `run_inference=false`、`create_submission=false`。
- Stage 1対応後の専用testは`13 passed`。local notebook実行は行わない。
- 全well raw identity preflightの初回実装では、exp404で固定したtyped-content SHA
  に対しCSV-text SHAを使ったため不一致を検出し、push前に停止した。
  exp404と同じcolumn名・dtype・bytes契約へ修正後、773 wellsと
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
  の一致を確認した。科学仕様、PF計算、入力ファイルは変更していない。
- Kaggle `kernels files`でexp209 HMM、exp226 reporting fold、
  exp115 hidden-like assignmentの正確なファイル名が各kernel outputに
  存在することを確認した。
- canonical version 1をpush前に`kaggle kernels pull -m`し、同じkernel id、
  id_no `129169339`、private / CPU / internet offを再確認した。
- Stage 1 package:
  - metadata: private / CPU / GPUなし / internetなし / run-on-push
  - dataset source: `kentookumura/exp404-v1-frozen-predictions`
  - kernel sources: exp209 / exp226 / exp115の3件
  - bootstrap flags: `run_stage_0=false`、`run_stage_1=true`、
    `run_inference=false`、`create_submission=false`
  - bootstrap source SHAとlocal source SHAは一致
  - canonical / compact notebookは27 cellsでcell source一致
- push直前validation:
  - 専用test: `13 passed`
  - Ruff: PASS
  - Jupytext変換: PASS
  - strict exp validation: PASS
  - template validation: PASS
- 2026-07-30 12:53:19 UTCに同じcanonical kernelへversion 2をpushした。
  - kernel:
    `kentookumura/exp483-huber-gr-filtering-likelihood-pf-train`
  - id_no: `129169339`
  - push直後status: running
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp483-huber-gr-filtering-likelihood-pf-train`

## 2026-07-30 Stage 1完了

- canonical kernel version 2は2026-07-30 16:23:19 UTCに`COMPLETE`となった。
- scope: `stage1_all_well_train_side_cv`
- status: `stage1_gate_failed_terminal_close`
- rows / wells / folds: `3,783,989 / 773 / 5`
- primary:
  - candidate RMSE: `11.095404595047105 ft`
  - 保存exp404 control RMSE: `10.914522073423171 ft`
  - improvement: `-0.1808825216239338 ft`
  - improved folds: `3 / 5`（必要`4 / 5`）
- fold改善量:
  - fold 0: `-0.24469602568233384 ft`
  - fold 1: `+0.09034776191547955 ft`
  - fold 2: `+0.08734729786473139 ft`
  - fold 3: `+0.1097953352143275 ft`
  - fold 4: `-0.8114743080780844 ft`
- 固定scope改善量:
  - raw GR observed: `-0.2533812630474852 ft`
  - raw GR missing: `-0.023391681559441935 ft`
  - high missing fraction: `+0.01222239822744342 ft`
  - MD since 1000+: `-0.20822805531889443 ft`
  - hidden-like spatial: `+0.14577831706492894 ft`
  - hidden-like typewell-purged: `-0.10956296819988687 ft`
- well-tail:
  - improved / worsened: `369 / 404`
  - delta RMSE p95: `+0.5209096346038877 ft`
  - worst: `70e1788b`、`+33.458522531259526 ft`
- fixed exp209 HMM/PF 50:50:
  - candidate / control: `10.162155357827986 / 10.084909848760013 ft`
  - candidate - control: `+0.07724550906797312 ft`
- technical gateは全PASS:
  - candidate PF wells / seed-well / particle starts:
    `773 / 98,944 / 49,472,000`
  - control PF / HMM / Beam / LightGBM / booster / GPU rerun: すべて`0`
  - raw identity、finite coverage、saved-control parity、fixed-blend parity、
    reporting folds、SHA readback、truth-late、runtime、RSS: すべてPASS
  - freeze前のtruth / control / fold / hidden-like role read: すべて`0`
- runtime:
  - prediction freeze: `12,361.117475748062 sec`
  - total: `12,454.353610515594 sec`
  - peak RSS: `3.5663185119628906 GB`
- scientific gateはFAIL。pooled、fold数、raw observed、raw missing、1000+、
  hidden-like typewell-purged、by-well p95/worst、fixed HMM/PF guardを満たさなかった。
- decision:
  `terminal_close_without_huber_or_pf_rescue`
- fold別・well別記録がKaggle logsだけでは不足したため、output archive全体ではなく
  primary/by-well/fixed-blend/gate/runtime/summary/auditの小さな成果物だけを取得した。
- scientific contract SHA:
  `089765cb14c395c1ff678d93c4a4940481aa7a8b846811287a7168ddda25d3c6`
- prediction logical / raw gzip / decompressed SHA:
  - `5a3c58aaaa1f9810cacc78836a87dc2ac06a8f2be614253d948b5a76056c3ad2`
  - `f805d83f4d6cc7e60a24d033333eebf1ddcfcb0e6209870d99c000a0831cad62`
  - `c0e2ea557d73eed0b463dfc4c4b17c3621cd3199a9d255e302e5ac35b91a274e`
- delta、scale、temperature、clip、mixture、particle/seed、transition、
  resampling、well/row gate、blend/selector、same-OOF rescueは行わない。
- inference、submissionは実行していない。
