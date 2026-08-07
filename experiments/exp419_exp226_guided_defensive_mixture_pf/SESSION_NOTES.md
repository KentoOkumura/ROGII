# exp419_exp226_guided_defensive_mixture_pf セッションノート

## 目的

exp226 geometryとlikelihood-PFをprediction blendやHMM residual-offsetとしてではなく、
importance-corrected defensive-mixture proposalとしてアルゴリズムレベルで統合する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_gate_failed_closed`
- scientific PF parent: exp404 x1.0 scale5
- geometry parent: exp226 fold-safe geometry-only OOF
- mechanism evidence: exp410
- negative reference: exp281
- candidate CV: `10.680074152793843`
- saved exp404 control CV / gain: `10.914522073423171 / +0.23444792062932862 ft`
- LB: なし
- candidate prediction: Kaggle merge v1 outputとして保存済み
- gate: technical PASS / mechanism FAIL / standalone adoption FAIL
- decision: `proposal_rejected_close_without_same_oof_rescue`
- ユーザー承認範囲: 正規train Notebook採用、preflight、4 shard、mergeまで完了。
  inference / submissionは未承認かつscientific gate不通過のため実行しない

## 変更点と固定判断

- exp226 absolute path、final、`gr_delta`、U projection、fixed offset stateは使わない
- fold-safe `tvt_geop + Z`の局所rateだけをproposal centerへ使う
- proposalは元transition 0.5、geometry中心の`1x / 4x / 16x`を各`1/6`
- rate importance ratio `p0/q`を掛け、clipしない。構成上`p0/q <= 2`
- target transition、position conditional、x1.0 GR emission、initialization、
  ESS resampling、roughening、500 particles、128 seedsは固定
- exp404 temperature-5 full-suffix evidence weightingを固定
- active scientific variantは1、保存controlを使いcontrol PFは再実行しない
- blend、HMM、adaptive proposal、donor-distance gate、MLは対象外

## 判定

mechanism gateはscale5比`>=0.10 ft`、4/5 folds、raw-GR observed、
missing / high-missing / 1000+ / hidden-like、exp410 support外率、固定episode SSE、
by-well tailをAND判定する。

standalone adoption gateはmechanism通過に加え、exp226 final比`>=0.03 ft`、
3/5 foldsを要求する。mechanismだけ通過した場合もinference資格は与えない。

## 実行量

- active scientific variants: 1
- candidate PF well-runs: 773
- control PF / exp226 / HMM / Beam well-runs: 0
- seeds per well / particles per seed: `128 / 500`
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- reporting folds: 5
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- GPU runs: 0
- Kaggle CPU shards: 4
- conservative / hard runtime per shard: `6h / 9h`

この固定実行量でpreflight、4 shard、mergeを完了した。mergeは保存artifactだけを読み、
scientific variant、PF well-run、model、GPUを追加していない。

## 再現性メモ

- seed:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- well内Numba single worker、shard順非依存
- stochastic要素はcomponent draw、rate / position noise、systematic resampling、
  roughening
- geometry weight 0 parity modeではcomponent drawを消費せずexp404 RNG順を保つ
- raw / decompressed / logical / schema / code / config / proposal contract /
  prediction SHAとKaggle kernel versionを記録済み
- candidateとtarget-free diagnosticsをfreezeしてからtruth / fold / scopeを結合する
- full coverage、全SHA、fixed-probe rerun parity前はdeterministic anchorと呼ばない

## コマンドログ

2026-07-27:

- `make new-steering EXP=exp419_exp226_guided_defensive_mixture_pf`
- `make new-exp EXP=exp419_exp226_guided_defensive_mixture_pf SOURCE=templates/experiment`
- steering、config、実験記録、backlogをdesign-onlyとして作成
- PF kernel / Jupytext / Notebook編集 / test / package / Kaggle実行は0

### 実装

ユーザーの`exp419を実装してください`を、設計済み境界のうちtrain-side実装承認として
受け取った。正規Notebook採用、Kaggle package、push、runは既存どおり別承認とした。

- `exp419_exp226_guided_defensive_mixture_pf_compact_selfcontained_train.py`
  をJupytext percent形式で実装し、compact候補Notebookへ変換した。
- exp404 exact kernelを維持し、rate samplingだけを元transition `0.5`、
  exp226 geometry中心の`1x / 4x / 16x`各`1/6`へ変更した。
- rate densityをlog spaceで評価し、GR likelihood前にclipなしの`p0/q`を掛ける。
  geometry weight 0 modeではcomponent drawを消費せず、exp404のRNG順を維持する。
- exp226 OOFはproposal前に`well_id / row_idx / suffix_offset / tvt_geop`だけを読む。
  `tvt_pred / gr_delta / tvt_true / error / abs_error / fold`はfreeze後だけ読む。
- 各shardでcandidate CSVに加え、各row × 128 seedのpre-GR predictive support
  min/maxをtarget-freeなfloat32 NPYとしてfreezeする。merge後にtruthを結合して初めて、
  majority-seed support外率を計算する。
- exp404 scale5 control、exp226 final、fold、hidden-like、exp410 target wells /
  839 fixed episodes / baseline row ledgerはcandidateとsupport SHA確定後だけ結合する。
- technical、mechanism、standalone adoptionの3 gateを実装した。mechanismだけPASSしても
  inference資格を与えない。
- 正規`*_train.ipynb` / `*_inference.ipynb`はtemplate placeholderのまま保持した。

### 実装時の固定実行量

- active scientific variants: 1
- candidate PF well-runs: 773
- control PF / exp226 / HMM / Beam rerun: 0
- seeds per well / particles per seed: `128 / 500`
- seed-well trajectories / particle starts: `98,944 / 49,472,000`
- reporting folds: 5
- LightGBM configs / trained folds / boosters / GPU: `0 / 0 / 0 / 0`

### 検証

- exp419専用test: `12 passed`
- geometry weight 0 synthetic fixture:
  exp404 prediction / log likelihood / resampling / ESS / clip countersがbitwise一致
- `jupytext --to ipynb --test`: PASS
- `py_compile`: PASS
- `ruff --select F821`: PASS
- `make validate-exp EXP=exp419_exp226_guided_defensive_mixture_pf`: strict PASS
- parent compact比較:
  exp404 `2,174`行 / exp419 `2,996`行。exp419は12章で、proposal、support freeze、
  4-shard merge、late truth、3 gateをNotebook上で追える。
- `task validate-exp`はこの環境に`task` executableがないため実行できず、
  Makefile同等コマンドでPASSを確認した。
- repo全体の`make test`はexp419 test実行前のcollectionで、既存のexp297 / exp301 /
  exp333 / exp336 / exp349 config contract不整合とexp411の既存Numba stub衝突の
  6 errorsにより停止した。exp419、`test_kaggle_notebooks.py`、
  `test_scaffold.py`をまとめた対象検証は`23 passed`である。

## 次のアクション

### 実行承認とpush前CPU枠確認

ユーザーの`実行してください`を、正規train Notebook採用、Kaggle package / push /
train-side runの承認として記録した。inference / submissionは未承認のままとする。

push前の固定実行量は次のとおりで、実装時契約から変更していない。

- active scientific variants: 1
- full candidate PF well-runs: 773
- saved control / exp226 / HMM / Beam full reruns: 0
- seeds per well / particles per seed: `128 / 500`
- seed-well trajectories / particle starts: `98,944 / 49,472,000`
- reporting folds: 5
- LightGBM configs / trained folds / boosters / GPU: `0 / 0 / 0 / 0`
- Kaggle CPU shards: 4
- full前technical probe: candidate 1 well + geometry-weight-zero control parity 1 well

設計に書かれたfull前1-well実artifact probeに対し、既存`probe` stageがfull merge後rerun
専用だった不足を修正した。`preflight_probe`はtruthを読まず、candidate / support / SHAを
freezeし、proposal allowlist、`p0/q <= 2`、保存exp404とのfloat32 parityを確認する。
full shard内の同一wellをpreflight candidateと比較し、再現性rerun証拠に使う。変更後の
exp419 + notebook/scaffold対象testは`24 passed`、Jupytext、py_compile、Ruff、
strict experiment validationはPASSした。

2026-07-27のpush前確認ではKaggle batch CPU上限は既存運用記録どおり5枠で、
RUNNINGは次の3本、空きは2枠だった。

- `kentookumura/exp411-pred-filtered-rate-innovation-destick-train`
- `kentookumura/exp402-foldsafe-grwr5-train-fold4-v2`
- `kentookumura/exp402-foldsafe-grwr5-train-fold1`

既存実行は停止しない。正規train Notebookへcompact候補を採用し、preflight packageを
作成する。実push直前にCPU RUNNING数を再確認し、1枠以上空いている場合だけ
canonical preflight kernelをpushする。

package作成後のpush直前再確認では、上記3本に次の2本が加わり、CPU上限5枠が
すべてRUNNINGになった。

- `kentookumura/exp416-rough-x10-shard-1`
- `kentookumura/exp416-rough-x10-shard-2`

このためexp419はpushしていない。検証済みpackageは
`kaggle/preflight/`へ凍結し、状態を`preflight_packaged_waiting_cpu_slot`とした。
既存5実行は停止せず、枠解放後に同じcanonical kernel ID/titleでpushする。

その後、ユーザー指示によりCPU枠の監視を停止した。exp419は未pushであり、
`preflight_packaged_monitoring_paused_waiting_user_notification`として保持する。
既存実行の完了連絡後に、CPU枠を再確認して同じpackage / canonical IDから再開する。

2026-07-28:

ユーザーから既存実行の完了連絡を受けて再開した。push前再確認ではRUNNINGは
`exp416-rough-x10-shard-0-v1`と`exp416-rough-x10-shard-3-v1`の2本で、
CPU上限5枠に対して3枠空いている。preflightは1 CPU枠だけを使うためpush可能と判断した。
昨日時点の未push packageは履歴として退避し、現在のconfigを埋め込んだpackageを
同じcanonical ID/titleで再生成・再検証してからpushする。

最初のpushはKaggle APIの詳細なし`SaveKernel 400`で拒否され、実行は開始しなかった。
指定したslug/title
`exp419-exp226-guided-defensive-mixture-pf-preflight`が51文字で、Kaggleの50文字上限を
1文字超えていたことが原因である。旧slugは`kernels pull -m`も403で、Notebookは
作成されていない。意味を保つ48文字のcanonical ID/title
`kentookumura/exp419-exp226-guided-defensive-mixture-preflight` /
`exp419 exp226 guided defensive mixture preflight`へそろえ、同じ実験・科学契約の
packageを再生成して再pushする。別実験や救済slugは作らない。

短縮canonical packageのidentity、24 tests、strict validationを再確認し、push直前にも
RUNNING 2 / 空き3枠を確認した。preflightはversion 1としてpush成功し、RUNNINGになった。

- kernel:
  `kentookumura/exp419-exp226-guided-defensive-mixture-preflight`
- Kaggle id_no / version: `128832139 / 1`
- runtime: CPU、GPU off、internet off
- package: `kaggle/preflight_v1/`
- inputs: competition 1、exp404 dataset 1、固定kernel sources 6

push後の`kernels pull -m`で同一ID、id_no、CPU metadata、inputsを確認した。
完了後はpreflight report / candidate / support SHAの実ファイル確認が必要なため、
この小さいoutputだけを取得してtechnical gateを判定する。

preflight version 1は約188秒でCOMPLETEし、ログと小さいprobe出力を取得した。
実ファイル検証結果は次のとおりでtechnical PASSである。

- status / stage: `passed / preflight_probe`
- probe well / rows: `01869cd4 / 5,557`
- importance ratio min / max: `0.0 / 1.999999991721266`
- geometry-weight-zero exp404 float32 parity max差: `0.0 ft`（閾値`1e-6 ft`）
- candidate logical SHA:
  `6fbacc32e7d27fbaf35ea83aca00b38d1501b8d26e319c73b47bab994c213907`
- support min/max: shape `(5,557, 128)`、finite、min<=max、保存SHA一致
- report raw SHA:
  `d8b5fba85859f17337496fa11cb791152df096f13c7c84fcf585db6e0b05852c`
- scientific contract SHA:
  `a25d809a7af142b74f3d7e5a8eec7f54247aa1bbf659b2a646493277fc50f013`
- proposal safe columns:
  `well_id / row_idx / suffix_offset / tvt_geop`
- forbidden / truth columns parsed: 0

preflight PASSを受け、固定1 variantのfull 4-shard packageへ進む。現在空き3枠のため、
shard0–2を先行pushし、shard3は空き確認後に同じ科学契約で起動する。

4 packageは同じsource SHA
`6bad3ed568d7ed7fb60707b2b1c2f8601b75e4c11a19f627a299f1511f2f5914`、
同じscientific contract SHA、shard indexだけが異なることを検証した。push直前に
exp418とexp413の別実験が開始され、RUNNING 4 / 空き1枠へ変わったため、
shard0だけをversion 1としてpushした。

- shard0 kernel:
  `kentookumura/exp419-exp226-guided-defensive-mixture-shard0`
- id_no / version / status: `128832582 / 1 / RUNNING`
- package: `kaggle/shard0_v1/`
- shard1–3: package検証済み、未push、CPU枠待ち

push後の`kernels pull -m`で同一ID、id_no、CPU metadataを確認した。既存実行は停止せず、
空きが出るたびに未push shardを1本ずつ起動する。

ユーザーから次の完了連絡を受けた後にCPU枠を再確認した。exp418はCOMPLETE、
exp416 shard0/3、exp413、exp419 shard0の4本がRUNNINGで、CPU上限5枠に対して
空きは1枠だった。この1枠だけを使ってshard1をversion 1としてpushした。

- shard1 kernel:
  `kentookumura/exp419-exp226-guided-defensive-mixture-shard1`
- id_no / version / status: `128832726 / 1 / RUNNING`
- package: `kaggle/shard1_v1/`
- shard2–3: package検証済み、未push、CPU枠待ち

push後の`kernels pull -m`で同一ID、id_no、CPU、GPU off、internet off、固定inputsを
確認した。これで既知のRUNNINGは5本、空きは0枠である。

その後、ユーザー指示によりshard2 / 3の実行を再開した。push前のCPU確認では、
exp416 shard0/3、exp413、exp419 shard0/1の5本がすべてCOMPLETEで、
CPU上限5枠に対して5枠空いていた。strict experiment validation、packageのcanonical
ID/title、`selected_stage: shard`、shard index、固定実行量を再確認し、2枠だけを使用して
shard2 / 3をversion 1としてpushした。

- shard2:
  `kentookumura/exp419-exp226-guided-defensive-mixture-shard2`
  （id_no `128912231`、RUNNING）
- shard3:
  `kentookumura/exp419-exp226-guided-defensive-mixture-shard3`
  （id_no `128912324`、RUNNING）

各pushの直前にCPU空きを確認した。push後の`kernels pull -m`で同一ID、id_no、CPU、
GPU off、internet off、固定inputsを確認し、`kernels status`で両方のRUNNINGを確認した。
shard0 / 1はCOMPLETE、shard2 / 3の完了後に4 shardをstrict mergeする。

2026-07-29:

shard2 / 3について、Kaggle CLI status、完了ログ、生成物一覧を再確認した。両方とも
tracebackなしで最終summaryが`status: complete`、candidate prediction、per-seed
predictive support、well audit / manifest、scientific contractが保存されている。
shard2は193 wells / 946,112 rows、shard3は194 wells / 945,732 rowsである。

4 shardが揃ったため、Kaggle outputをローカルへ丸ごと取得せず、各version 1 Notebookを
merge Notebookのkernel inputとして直接接続する。mergeは保存済み4 shard、
preflight、exp404 control、exp226 geometry、exp410 ledgers、exp115 assignmentを読み、
strict coverage / SHA / contract / probe rerun parityを検証してからtruthをlate attachし、
technical / mechanism / standalone adoption gateを判定する。

merge push前の追加実行量は次のとおり。

- scientific variants / candidate PF well-runs: `0 / 0`（保存済み4 shardのみ使用）
- parent control / exp226 / HMM / Beam reruns: `0 / 0 / 0 / 0`
- LightGBM configs / trained folds / boosters / GPU: `0 / 0 / 0 / 0`
- Kaggle CPU Notebook: `1`
- canonical kernel:
  `kentookumura/exp419-exp226-guided-defensive-mixture-merge`
- package予定: `kaggle/merge_v1/`

`execution.selected_stage`を`merge`へ切り替え、4 shard rootを順序固定し、preflightと
4 shardをkernel sourcesへ追加した。packageを生成・検証後、push直前にCPU枠を確認する。

merge packageはcanonical ID/title、11 kernel sources、1 dataset source、
1 competition source、固定4 shard順序、source SHA一致を確認した。Jupytext test、
py_compile、Ruff F821、exp419専用`13 passed`、strict experiment validationはすべて
PASSした。

push直前に直近15 Notebookのstatusを確認し、RUNNINGは
`kentookumura/exp429-self-gr-weak-boost-likpf-ablation-train`の1本だけで、
CPU上限5枠に対して4枠空いていた。既存実行を停止せず、mergeをversion 1としてpushした。

- kernel:
  `kentookumura/exp419-exp226-guided-defensive-mixture-merge`
- id_no / version / status: `128974840 / 1 / RUNNING`
- package: `kaggle/merge_v1/`
- runtime: CPU、GPU off、internet off
- inputs: competition 1、exp404 dataset 1、kernel sources 11

push後の`kernels pull -m`で同一ID、id_no、runtime、全inputsを確認し、
`kernels status`でRUNNINGを確認した。

merge version 1はKaggle上でCOMPLETEした。strict mergeとlate truth attachmentを
`152.088465秒`で完了し、最終状態は
`train_side_guided_defensive_mixture_gate_failed_closed`である。outputは
`kaggle/output/merge_v1/`へ取得し、manifest記載のraw / decompressed SHA、
3,783,989 data rows、gate JSON、metrics JSONを実ファイルで確認した。

technical gateはPASSした。

- coverage: `3,783,989 rows / 773 wells / 5 folds`、finite 1.0、fallback 0
- execution: 1 variant、773 PF wells、98,944 seed-well、
  49,472,000 particle starts、control / exp226 / HMM / Beam / model / GPU rerun 0
- proposal allowlist: `well_id / row_idx / suffix_offset / tvt_geop`
- freeze前truth / fold / hidden-like / control / exp226 final / exp410 scope read: 0
- importance ratio min / max: `0.0 / 1.999999999999981`
- geometry-weight-zero parity: `0.0 ft`
- preflight / full fixed-well prediction: float32 byte-identical
- saved control RMSE parity差: `0.0 ft`

scientific結果は次のとおり。

- candidate RMSE: `10.680074152793843`
- saved exp404 RMSE: `10.914522073423171`
- exp404比gain / improved folds: `+0.23444792062932862 ft / 4 of 5`
- raw-GR observed gain: `+0.16554386971295187 ft`
- persistent episode SSE reduction: `14.82130574067695%`
- candidate support外率 / exp410 baseline:
  `97.49733915302067% / 64.20611555399323%`
- support外率のreduction: `-33.29122359902744 percentage points`
- hidden-like spatial gain: `-0.1158233436125844 ft`
- by-well delta RMSE p95 / worst: `+5.766212571731278 / +20.570237512061006 ft`
- saved exp226 final RMSE: `9.427109596582222`
- exp226比gain / improved folds: `-1.2529645562116212 ft / 1 of 5`

technical gateはPASS、mechanism gateとstandalone adoption gateはFAIL、
`mechanism_positive: false`である。pooled RMSE、4 folds、raw-GR observed、
episode SSEは改善したが、主目的のfinite-support外率が33.2912 points悪化し、
hidden-like spatialとwell-tail guardも大幅にFAILした。importance correctionで
target posteriorを保っても、500粒子の半数をgeometry proposalへ割くことが
有限粒子supportを壊したと解釈する。

prediction logical / raw gzip / decompressed SHAは順に次である。

- `2465d2aae907af57ad16daa0588d3210b8d201bad5f622f4408f5f4d3b701740`
- `0165104cc606c1a5d64f7682f9ae1ad946f8b5e490c0efc15bf8a10c06887789`
- `8b7e8fc05a4cf529d0f9d4fe1cab8fc041c30eeb88e555e55f0aaf2c4255ea43`

artifact manifest SHAは
`fa7a2be4d494edf9813c56647fc52f8fba02f7697bcf3e2c14d1967bac66ce0b`。
事前登録どおり
`proposal_rejected_close_without_same_oof_rescue`とし、mixture weight / width、
importance clip、GR sigma、process noise、roughening、seed / particle数、
well / row gateを同じOOFで探索しない。inference / submissionへ進めない。
