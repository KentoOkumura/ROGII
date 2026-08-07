# exp484_student_t_gr_filtering_likelihood_pf セッションノート

## 目的

exp374 Student-t emissionをexp404/417 PF filtering likelihoodへ移植し、
fixed32 Stage 0を通過した固定候補を全773 wellsのStage 1で評価する。

## 現在の状態

- Route: `pf_beam`
- Status: `stage1_gate_failed_terminal_close`
- Priority: P2
- implementation / contract test / 正規train Notebook: 実装済み
- Kaggle package / push / Stage 0実行: kernel version 2で完了
- Stage 1: canonical kernel version 3で`COMPLETE`
- inference / submission: gate FAILにより未実行
- CV: `10.89709692312777`
- LB: なし

## 根拠

- exp374 direct gain `+0.217808533 ft`、4/5 folds。
- by-well p95 `+0.982661344 ft`、worst `+35.015963236 ft`でterminal FAIL。
- PF particle likelihoodとしては未検証。

## 実行契約

- Stage 0: 32 PF wells、4,096 seed-well、2,048,000 particle starts。
- Stage 1: 773 PF wells、98,944 seed-well、49,472,000 particle starts。
- control PF / HMM / Beam / model / booster / GPU rerun 0。
- Stage 0実績は上記どおり。Stage 1ではcandidate 1 variantだけを追加実行する。

Stage 0 fixed32はwell IDのみから
`sha256("exp484::stage0::<well_id>")`を計算した先頭32 wellsで固定した。
165,010 suffix rowsで、truth / error / fold / hidden-like roleはmanifestに
含めない。

## 再現性

exp404 stable seeds、truth-late、fixed order、content SHAを継承する。
variant名はseed keyに入れない。保存exp404 controlはcandidate prediction、
schema、logical/decompressed SHA、well auditをfreezeした後だけ読み、
control PFを再実行しない。

## 実装ログ

- 2026-07-30:
  - 追加依頼`exp484を実装してください`をimplementation承認として記録。
  - `exp484_student_t_gr_filtering_likelihood_pf_compact_selfcontained_train.py`
    をJupytext percent形式で作成し、正規train Notebookへ採用。
  - exp404 kernelのper-particle emissionだけを
    `-0.5 * (df + 1) * log1p(z^2 / df)`、`df=4`へ置換した。
  - state-independent normalization constantは省略し、追加z/log clipは
    入れていない。exp404由来の数値likelihood floor `1e-300`は維持。
  - dynamics、500 particles、128 seeds、x1.0 GR scale、T=5 aggregation、
    ESS resampling、roughening、missing-GR interpolationを固定。
  - formula、中心2次近似、extreme residual finite positive weight、
    exp404 x1.0 input parity、stable seed、truth-late、SHA、fail-closed
    inference guardの専用testを追加。
  - Stage 0時点の親compactとの比較:
    - exp404 compact train: 2,174行、主要章11個。
    - exp484 compact train: 1,968行、主要章10個。
    - input preparation、Student-t PF kernel、fixed32 selection、prediction
      freeze、saved-control/truth-late、technical gate、generated artifactsを
      Notebookセルに展開しており、同一exp helper importだけの薄い構成ではない。
  - 実行した検証:
    - `.venv/bin/python -m py_compile ...`
    - `.venv/bin/ruff check ... --select F821,F401,E9`
    - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`
    - `.venv/bin/pytest -q tests/test_exp484_student_t_gr_filtering_likelihood_pf.py`
  - 専用testは`10 passed`。Kaggle package、Stage 0/1実行、prediction、
    inference、submissionは0。
  - 保存exp404 control実ファイル3,783,989行についてraw SHA
    `b3699432...90f8`、decompressed SHA `00fe1b90...0d00`を実照合した。
    pre-serialization logical SHAはCSVから再構成せずprovenanceとして保持し、
    executable input guardはraw/decompressed SHA、固定32 subsetには新しい
    logical SHAを付与する。
  - `task validate-exp`はTask CLI未導入のため実行不可だったので、
    `make validate-exp EXP=exp484_student_t_gr_filtering_likelihood_pf`を実行し
    strict validation PASS。`make validate-template`もPASS。
  - notebook/scaffold testは`11 passed`。
  - 追加依頼`実行してください`をcanonical Kaggle CPU package作成、
    push、fixed32 Stage 0実行の承認として記録した。
  - 実行対象はStudent-t `df=4`の1 variant、32 PF wells、
    4,096 seed-well trajectories、2,048,000 particle starts。
    control PF / HMM / Beam / LightGBM / GPU再実行は0。
  - Stage 1、inference、submissionは引き続き無効で、別承認が必要。
  - canonical train packageを次の設定で生成した:
    - kernel ID:
      `kentookumura/exp484-student-t-gr-filtering-likelihood-pf-train`
    - title: `exp484 student t gr filtering likelihood pf train`
    - notebook SHA256:
      `de682253cdb0e2591505ce0d34bb82b7d7ccc3c83b5f39ab51d33a414b1d47a1`
    - private / CPU / internet off / run-on-push / competition sourceあり
    - saved exp404 dataset sourceあり、fixed32 manifest bootstrap SHA一致
  - Kaggle kernel version 1をpush。Kaggle側pullで`id_no=129170461`、
    canonical ID/title、`enable_gpu=false`、`enable_internet=false`、
    competition/dataset sourcesを確認した。
  - v1は開始約34秒後、candidate生成前のraw identity preflightで
    `ValueError: exp484 raw train well identity mismatch`となりERROR。
    別slugへのpushや科学条件変更は行っていない。
  - 原因はexp484の`dataframe_content_sha`だけがCSV再直列化方式で、
    親exp404のcolumn/dtype/bytes方式と異なっていた実装バグ。
    ローカル773 wellsでexp484旧方式`f0dec823...e422`、
    exp404方式`bbb687a1...9b32`を実照合し、後者がconfig期待値と一致した。
  - SHA関数をexp404と完全同一へ修正し、親方式との回帰testを追加した。
    Student-t emission、PF dynamics、seed、対象well、実行規模は変更していない。
  - 修正後は専用test `11 passed`、実データ773 wells identity
    `bbb687a1...9b32`一致、Jupytext round-trip、strict validation PASS。
  - 同じcanonical IDへkernel version 2をpushした。v2 package notebook SHA256は
    `bb18defe4454121351ffaf10bfd0dbe4f30ad2400a14ab4feb01bf9f7599227a`。

## 実行前コスト確認

- active scientific variant: 1
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- Stage 0 candidate PF well-runs: 32
- seeds / particles: `128 / 500`
- seed-well trajectories / particle starts: `4,096 / 2,048,000`
- 保存exp404 control PF rerun: 0
- HMM / Beam / GPU: `0 / 0 / 0`
- Stage 1は`run_stage_1: false`で、Stage 0全PASS後も別承認が必要。

## Stage 0完了

- canonical kernel version 2は2026-07-30 12:53:34 UTCに`COMPLETE`。
- status: `stage0_passed_pending_separate_stage1_approval`
- rows / wells: `165,010 / 32`
- 実行量:
  - scientific variant: `1`
  - candidate PF well-runs: `32`
  - seed-well trajectories: `4,096`
  - particle starts: `2,048,000`
  - control PF / HMM / Beam / LightGBM / booster / GPU: すべて`0`
- technical gate: `16 / 16 PASS`
  - formula、fixed df=4、x1.0 scale、finite coverage
  - stable seed、ESS / resampling ledger、execution count
  - raw / fixed32 / saved-control SHA、prediction / audit SHA
  - truth/control/fold/hidden-likeのfreeze前read 0
  - runtime projection、peak RSS
- fixed32 truth-late diagnostic（CV / promotion evidenceではない）:
  - candidate RMSE: `16.536326062515673 ft`
  - 保存exp404 control RMSE: `17.358983592771203 ft`
  - control - candidate gain: `+0.8226575302555297 ft`
  - improved wells: `18 / 32`
  - by-well delta p95: `+0.3870566934830214 ft`
  - worst-well delta: `+0.7826119005022605 ft`
- runtime:
  - candidate: `629.2169744968414 sec`
  - total: `667.7507383823395 sec`
  - full 773 projection: `15199.522540189326 sec`
  - peak RSS: `0.9093971252441406 GB`
- reproducibility:
  - scientific contract:
    `af07896332346cccf722bcedc1cee5c371d93089e9fab2a49e19cada2cb5cc36`
  - prediction logical:
    `9527bdff3feb26c86a4943a6d4f48dc1fb1d4a9fe9f33fdc3f5d3071579cf483`
  - prediction raw gzip:
    `6543268e88f3792ba965d053150ae07c67863e690612b44bb28680c23f039781`
  - prediction decompressed:
    `8b20ad290de96f0e2d972c40025be3fa7f509a8594e5a8714b04c0fafa02edae`
  - well audit:
    `deefa093b64b7a7a03d6851f840612f85f9e985a33679f9ac1c5958dd33f721e`
- metrics、technical gate、prediction、truth-late readout、input/by-well/auditを
  SHA実照合のため`kaggle/output/train_v2/`へ選択取得した。
- local Notebook実行、Stage 1、inference、submissionは行っていない。
- 完了記録反映後のcompact / canonical train Notebook SHA256は一致:
  `9c00f355eae9adce647c1d4ee2b0fe8908494a135ff4ecfc64cd5bb139f34f44`。
- 最終検証は専用test `11 passed`、`py_compile`、Ruff、
  Jupytext round-trip、strict experiment validation、template validation、
  `metrics.json` parseがすべてPASS。

## 次のアクション

Student-t filtering-likelihood branchは事前登録どおりterminal closeとする。
df/scale/temperature/clip/mixture/particle/seed/dynamics、well/row gate、
blend/selector、same-OOF rescue、inference、submissionへ進まない。

## 2026-07-30 Stage 1実行承認

- ユーザー依頼`Stage1へ進んでください`を、同じ科学契約の全773 wells
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
- Stage 1はStudent-t `df=4`、x1.0 GR scale、500 particles、
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
- Kaggle `kernels files`でexp209 HMM、exp226 reporting fold、
  exp115 hidden-like assignmentの正確なファイル名が各kernel outputに
  存在することを確認した。
- canonical version 2をpush前に`kaggle kernels pull -m`し、同じkernel id、
  id_no `129170461`、private / CPU / internet offを再確認した。
- compact / canonical train Notebookは24 cellsでcell source一致、
  SHA256は
  `60edd7b6525df56754de2c81e350269415541d301e81241071df1420f801763d`。
- Stage 1 package:
  - metadata: private / CPU / GPUなし / internetなし / run-on-push
  - dataset source: `kentookumura/exp404-v1-frozen-predictions`
  - kernel sources: exp209 / exp226 / exp115の3件
  - bootstrap flags: `run_stage_0=false`、`run_stage_1=true`、
    `run_inference=false`、`create_submission=false`
  - bootstrap source SHAとlocal source SHAは
    `51ec23a7508932e6bc2350efa4ab87f17752ae4c5097dff635dd3f39899212c7`
    で一致した。
- push直前validation:
  - 専用test: `13 passed`
  - Ruff: PASS
  - Jupytext変換: PASS
  - strict exp validation: PASS
  - template validation: PASS
- 2026-07-30 13:22 UTCに同じcanonical kernelへversion 3をpushした。
  - kernel:
    `kentookumura/exp484-student-t-gr-filtering-likelihood-pf-train`
  - id_no: `129170461`
  - package notebook SHA256:
    `57fb941c25ee837aafbe9e9fe894bf961ea5ba094c18d70771669d53f5ce0aa9`
  - push直後status: `RUNNING`
  - Kaggle側metadata: private / CPU / internet off、exp404 dataset、
    exp209 / exp226 / exp115 kernel sourcesを確認
  - URL:
    `https://www.kaggle.com/code/kentookumura/exp484-student-t-gr-filtering-likelihood-pf-train`

## 2026-07-30 Stage 1完了

- canonical kernel version 3は2026-07-30 16:25:05 UTCに`COMPLETE`となった。
- scope: `stage1_all_well_train_side_cv`
- status: `stage1_gate_failed_terminal_close`
- rows / wells / folds: `3,783,989 / 773 / 5`
- primary:
  - candidate RMSE: `10.89709692312777 ft`
  - 保存exp404 control RMSE: `10.914521913422574 ft`
  - improvement: `+0.017424990294804488 ft`（必要`+0.05 ft`）
  - improved folds: `2 / 5`（必要`4 / 5`）
- fold改善量:
  - fold 0: `-0.10465642717847423 ft`
  - fold 1: `-0.14741840814164675 ft`
  - fold 2: `-0.07724560226002275 ft`
  - fold 3: `+0.08487579725820638 ft`
  - fold 4: `+0.27548005045857415 ft`
- 固定scope改善量:
  - raw GR observed: `-0.06890035663986005 ft`
  - raw GR missing: `+0.20536830388171623 ft`
  - high missing fraction: `+0.1651920417417685 ft`
  - MD since 1000+: `+0.028529855371738577 ft`
  - hidden-like spatial: `+0.1299765870556122 ft`
  - hidden-like typewell-purged: `-0.1301462559061548 ft`
- well-tail:
  - improved / worsened: `442 / 331`
  - delta RMSE p95: `+1.4550666558525527 ft`
  - worst: `d924e971`、`+16.664889732705497 ft`
- fixed exp209 HMM/PF 50:50:
  - candidate / control: `10.067803689497714 / 10.084909779429955 ft`
  - improvement: `+0.017106089932241275 ft`、guard PASS
- technical gateは`18 / 18 PASS`:
  - candidate PF wells / seed-well / particle starts:
    `773 / 98,944 / 49,472,000`
  - control PF / HMM / Beam / LightGBM / booster / GPU rerun: すべて`0`
  - formula、stable seed、ESS/resampling、raw identity、finite coverage、
    saved-control parity、fixed-blend parity、reporting folds、SHA、
    truth-late、runtime、RSS: すべてPASS
  - freeze前のtruth / control / fold / hidden-like role read: すべて`0`
- runtime:
  - prediction freeze: `10,871.425777435303 sec`
  - total: `10,959.720402002335 sec`
  - peak RSS: `3.30963134765625 GB`
- scientific gateはFAIL。pooled gain、fold数、raw observed、
  hidden-like typewell-purged、by-well p95/worstを満たさなかった。
- decision:
  `terminal_close_without_student_t_or_pf_rescue`
- fold別・well別記録がKaggle logsだけでは不足したため、output archive全体ではなく
  primary/by-well/fixed-blend/gate/runtime/summary/auditの小さな生成物だけを
  `kaggle/output/train_v3/`へ選択取得した。
- scientific contract SHA:
  `af07896332346cccf722bcedc1cee5c371d93089e9fab2a49e19cada2cb5cc36`
- prediction logical / raw gzip / decompressed SHA:
  - `4dbe939363b1522dbc521680cd25d3ce7993ff8b94ddbdbf30b95073db2b28f4`
  - `e1344c2dc63dda8905b997bb82b8c3e25ea9df3cfade2688e2e48dab9f8cc655`
  - `1b02c68b0e52cd031b9d06e931f8fa92e2fb841dbc9e4ee6c7778807af5f1962`
- df、scale、temperature、clip、mixture、particle/seed、transition、
  resampling、well/row gate、blend/selector、same-OOF rescueは行わない。
- inference、submissionは実行していない。
- 完了状態を反映したcompact / canonical train Notebook SHA256は一致:
  `ff828ab8b4309593131b6091961f9de97787ae787e85c15d4e6a2d091560e4a7`。
- 最終検証は専用test `13 passed`、`py_compile`、Ruff、Jupytext
  round-trip、strict experiment validation、template validation、
  Stage 1 metrics / config / fold / SHA consistencyがすべてPASS。
