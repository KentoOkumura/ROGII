# exp446_persistent_tvt_rate_exact_hmm セッションノート

## 目的

exp209の持続`U-rate`状態を持続`TVT-rate`状態へ置換する仮説を、
反証可能な単一candidateとして固定し、fixed32 Stage 0でmechanismを評価する。

Assumption: ユーザーの「その実験」は、exp445のparity-onlyな再ラベルではなく、
`(TVT, persistent TVT-rate)`のscientific treatmentを指す。

## 現在の状態

- Route: `pf_beam`
- 状態: fixed32 Stage 0完了、`stage0_fail_closed`
- 優先度: 低-中・P3
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- CV / LB: なし / なし
- 実装: 2026-07-30のユーザー依頼で承認され完了
- 正規Notebook採用、Kaggle package、fixed32 Stage 0:
  2026-07-30のユーザー依頼「実行してください」で承認
- Stage 1、rerun、inference、submission: fail-closed

## 設計根拠

- exp408: persistent offsetの主因はforward transition/prior hysteresis
  `59.3978%` SSE。rateの0方向under-responseは`70.9074% rows /
  70.3580% SSE`。
- exp435: rate履歴を除去したTVT-only HMMはmatched controlを大幅に悪化させたが、
  persistent TVT-rateは未検証。
- exp441: U-rateの全support化だけではunder-response削減が2.297 pointsに留まり、
  forward/persistent SSEも改善しなかった。
- exp445: TVT positionとrow-shifted U positionの座標パリティは検証済みだが、
  rate dynamicsはU-rateのままで性能仮説ではない。

## 固定した差分

- parent: `r_U=d(TVT+Z)/dMD`、
  `delta_TVT=r_U*delta_MD-delta_Z`。
- candidate: `q=dTVT/dMD`、
  `delta_TVT=q*delta_MD`。
- candidate prefix rate:
  `median(delta_TVT_input/delta_MD)`。
- candidate grid:
  `linspace(-span_q,span_q,41)`、
  `span_q=max(0.10,abs(q0)+0.04)`。
- exp209から固定:
  momentum `0.998`、`sig_r=0.002`、`sig_p=0.02`、position step
  `0.35 ft`、GR emission、prior、posterior readout。
- scientific variant: 1。

## 実行契約

- Stage 0: fixed32、候補32 HMM well-runs、parent rerun 0。
- Stage 1: Stage 0全AND gate PASS・別承認時だけ候補773、parent rerun 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- fixed32はmechanism-onlyでCVではない。
- rate/grid/noise/emission/prior/gate/blend/selectorの救済は禁止。

## コマンドログ

```bash
make new-steering EXP=exp446_persistent_tvt_rate_exact_hmm
make new-exp EXP=exp446_persistent_tvt_rate_exact_hmm
```

- 2026-07-30: steeringを先に作成し、その後テンプレートから実験scaffoldを作成。
- 親実験のコードやNotebookはコピーしていない。
- config、README、SESSION_NOTES、result、metricsだけをdesign-onlyへ更新。
- このdesign-onlyセッション時点ではcandidateコード、test、
  Jupytext source、Kaggle packageは未作成だった。

```bash
make validate-template
make validate-config
make validate-exp EXP=exp446_persistent_tvt_rate_exact_hmm
make update-summary
```

- project template validation: PASS。
- project strict config validation: PASS。
- exp446 strict validation: PASS。
- config / metrics / route / status / authorization / 実行量の
  design consistency assertion: PASS。
- `backlog/KAGGLE_DIRECTION.md`未着手backlogと`experiment_summary.md`へ反映済み。
- train / inference notebookは各6 cellsのテンプレートscaffoldであり、
  candidate HMM実装はない。

## 2026-07-30 compact self-contained実装

ユーザー依頼:

```text
exp446を実装してください
```

承認範囲はcompact候補と専用testまでと解釈した。AGENTS.mdの既存Notebook
非上書き規則、および事前設計の承認境界に従い、正規
`exp446_persistent_tvt_rate_exact_hmm_train.ipynb` /
`..._inference.ipynb`は変更していない。Kaggle package、push、Stage 0/1、
inference、submissionも行っていない。

作成:

- `exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py`
- `exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.ipynb`
- `exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py`
- `exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.ipynb`
- `experiments/exp446_persistent_tvt_rate_exact_hmm/tests/test_exp446_persistent_tvt_rate_exact_hmm.py`

実装内容:

- prefix末尾50 rowsの有効stepから
  `q0=median(delta_TVT_input/delta_MD)`を計算する。
- `span_q=max(0.10,abs(q0)+0.04)`、41 statesのrate gridを作る。
- exp209と同じ隣接3-bin Euler kernel、`momentum=0.998`、
  `sig_r=0.002`をq gridへ適用する。
- arrival rateのposition meanを`q_destination*delta_MD`とし、
  candidate transitionから`delta_Z`を除外する。
- TVT grid、GR emission、position prior、position kernel、
  forward/backward、posterior mean/stdはexp209/exp441構成を維持する。
- candidate actual pathは保存exp209 predictionだけをcontrolとして読み、
  parent HMMを再実行しない。
- constant-Z synthetic sentinelだけparent `U-rate` pathを実行し、
  rate grid/kernel、position kernel、log likelihood、position/rate
  posterior、TVT mean/stdの数値一致を確認する。
- small-state dense forward/backward reference、rate/position normalization、
  q-position mean formula、truth-late ledgerを実装する。
- input、rate grid、rate kernel、joint transition、posterior、prediction、
  diagnostic、metricsのSHAを保存する。gzipはdeterministic gzipとし、
  decompressed logical contentのreadback一致をgateする。
- exp408 parent rate readoutは
  `parent_filtered_U_rate-delta_Z/delta_MD`へ変換し、candidate qと同じ
  TVT-rate座標でzero-directed under-responseを比較する。

構成比較:

- direct parent exp209にはcompact self-contained版がないため、同じfixed32
  truth-late exact-HMM系のexp441とtechnical parity系のexp445を参照した。
- line countはexp441 `3070`、exp445 `2704`、exp446 `3445`。
- exp446は10章で、runtime/config、fixed32/saved parent、TVT-rate input、
  local kernels、exact HMM/sentinel、target-free freeze、truth-late readout、
  gates、guarded orchestrationをNotebook上に展開した。
- 同一exp helper import、`__file__`、`Path(__file__)`は0。

検証:

```bash
.venv/bin/pytest -q experiments/exp446_persistent_tvt_rate_exact_hmm/tests/test_exp446_persistent_tvt_rate_exact_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py
.venv/bin/ruff check \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_train.py \
  experiments/exp446_persistent_tvt_rate_exact_hmm/exp446_persistent_tvt_rate_exact_hmm_compact_selfcontained_inference.py \
  experiments/exp446_persistent_tvt_rate_exact_hmm/tests/test_exp446_persistent_tvt_rate_exact_hmm.py --select F821
make validate-template
make validate-config
make validate-exp EXP=exp446_persistent_tvt_rate_exact_hmm
```

- dedicated tests: `12 passed`。
- train/inference Jupytext round-trip: PASS。
- py_compile / Ruff F821: PASS / PASS。
- project template / strict config / strict experiment validation:
  PASS / PASS / PASS。

GPUコストガード:

- active scientific variant: 1。
- Stage 0 candidate: 1 variant × 32 wells = 32 HMM well-runs。
- Stage 1上限: 1 variant × 773 wells = 773 HMM well-runs。
- 保存exp209 parent rerun: Stage 0/1とも0。
- LightGBM config / fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- compact実装セッション時点の実行は0であり、その後のStage 0実行前に
  実行量を再確認して別承認を得た。

## 2026-07-30 fixed32 Stage 0実行承認とpush前再確認

ユーザー依頼:

```text
実行してください
```

この依頼を、正規Notebook採用、Kaggle package作成、Kaggle private CPUでの
固定Stage 0を1回実行する承認として扱う。Stage 1、inference、submissionは
承認範囲に含めない。

push前の実行量:

- active scientific variant: `persistent_tvt_rate` 1本。
- Stage 0: 1 variant × fixed32 wells = 32 candidate HMM well-runs。
- 保存exp209 controlを読み、parent HMM rerunは0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- PF / Beam / GPU: `0 / 0 / 0`。
- Stage 1の773 HMM well-runsは実行しない。
- inference / submissionは実行しない。

Kaggle実行条件:

- kernel: `kentookumura/exp446-persistent-tvt-rate-exact-hmm-train`
- title: `exp446 persistent tvt rate exact hmm train`
- private CPU、GPUなし、internetなし、1 worker / Numba 1 thread。
- 入力はcompetition data、保存済みexp209 prediction kernel、
  exp408 row-ledger kernel、bootstrap内fixed32/episodeファイル。
- fixed32はmechanism preflightでありCVまたはpromotion結果として扱わない。
- 全technical/mechanism gate PASSでもStage 1は自動実行せず、別承認を得る。

正規Notebook採用とpackage検証:

- compact self-contained train/inferenceを同名の正規Notebookへ採用した。
- train Notebookは25 cellsで、scaffoldの`Metrics scaffold`を含まず、
  exact HMM、constant-Z sentinel、fixed32 orchestrationをセル内に展開した。
- package metadataのid/title slug一致、private、CPU、internet off、
  run-on-push、competition source、exp209/exp408 kernel sourcesを確認した。
- canonical adoption後の専用testは`12 passed`。Jupytext round-trip、
  py_compile、Ruff F821、strict exp validationもPASSした。

初回push:

- `kentookumura/exp446-persistent-tvt-rate-exact-hmm-train`へpushしたが、
  Kaggle `SaveKernel 400 Bad Request`で実行開始前に拒否された。
- 直前の同slug metadata pullは403で、kernel作成済みとは確認できなかった。
- id/titleは同じ42文字相当のslugへ解決され、50文字上限内で一致している。
- package notebookは既定のrepository `src/`を含み約1.3 MiBだった。正規trainは
  self-containedで`src` importや`sys.path`依存がないため、既存のexp440/443と
  同じAPI request縮小として`--no-src`で再packageする。
- 科学条件、config、fixed assets、kernel sources、canonical id/title、
  実行量は変更しない。旧pushではkernel version・HMM runとも0。

`--no-src`再packageとStage 0 push:

- package notebook: `955,724 bytes`
- package notebook SHA:
  `f0a31dacf3085e67e4b3348c2eabac06832e9bf371f6f1cf57ba25bad784631c`
- metadata SHA:
  `722cff9f24c7363eadb3f7768751971bed2517b3a268a184488503681c087bb3`
- pushed package config SHA:
  `22962bdd777b039249a952e262bc32361b83da4549a34c966ae269ac1664969a`
- 専用test `12 passed`、strict validation PASS後、同じcanonical slugへの
  pushが成功した。
- kernel:
  `kentookumura/exp446-persistent-tvt-rate-exact-hmm-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp446-persistent-tvt-rate-exact-hmm-train`
- version: `1`
- id_no: `129106260`
- push: `2026-07-30 00:01:38 UTC`
- private CPU、internet off、Stage 0実行中。
- push後pullで同一id/title、private、GPU/internet off、competition source、
  exp209/exp408 kernel sourcesを確認した。
- push後のローカルconfigは監視情報だけを追記しており、実行中version 1の
  package config SHAは上記値で固定する。監視記録だけを理由に再pushしない。

## 2026-07-30 Kaggle Stage 0 version 1結果

- Kaggle status: `KernelWorkerStatus.COMPLETE`
- scientific candidate / HMM wells / suffix rows:
  `1 / 32 / 156,088`
- reporting folds: 5
- 保存exp209 parent rerun / LightGBM config / trained fold / booster /
  fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`
- elapsed: `1,928.728804457 sec`
- peak RSS: `1.131355286 GB`
- fixed32はmechanism-onlyで、CV / promotion evidenceではない。
- Kaggle output archiveは取得していない。terminal logsに全gate値、
  生成物パス、行数、logical/decompressed SHA、kernel情報が含まれるため、
  AGENTS.mdのCV/output取得方針に従いlogsを正の根拠とした。

Technical gateは`17 / 18 PASS`:

- finite coverage `1.0`、normalization max error `3.642e-14`。
- constant-Z parent parityはkernel/posterior/prediction差`0.0`。
- small-state dense reference prediction差`4.042e-09`。
- position edge residual `0.0 ft`。
- truth/role/fold/episodeのfreeze前read `0`。
- peak RSS `1.131355 <= 25 GB`。
- full 773-well runtime projectionだけ
  `46,590.855183 > 30,600 sec`でFAIL。

Mechanism gateは`0 / 7 PASS`:

- parent / candidate zero-directed under-response SSE share:
  `0.242821116 / 0.303912103`。
- share削減絶対値:
  `-0.061090987 < +0.05`、悪化。
- forward-cause episode SSE削減:
  `-0.306441222 < +0.10`、悪化。
- persistent episode SSE削減:
  `-0.214831092 < +0.05`、悪化。
- persistent改善well:
  `5 / 16 < 10 / 16`。
- persistent改善fold:
  `2 / 5 < 4 / 5`。
- matched control pooled RMSE delta:
  `+7.159063411 > +0.02 ft`。
- matched control by-well delta p95:
  `+16.310621759 > +0.25 ft`。

主要SHA:

- scientific contract:
  `99ab27aa50ecc38a20f10ae39d8709f55bba3323c28fe9c2b036b2cf417659f1`
- transition / rate-grid / rate-kernel manifests:
  `5813d4de...49cc3 / 43b7f793...17b30 / 9be559f0...8e7c0`
- posterior / prediction / diagnostic manifests:
  `34c94c88...407d9 / 35581c23...931e / c01f4ffe...255a4`
- prediction decompressed:
  `72c40bd73b71e469b49abdb25b0eb3048150b37e3f9624a94787338ad9eb3634`
- diagnostic decompressed:
  `4ad24705604b3895278910ca875b9240fc4f7a2766a0cf6eecc2472e8ad5b4dc`

判断:

- technical runtimeと全mechanism gateがFAILし、Stage 1 eligibleはfalse。
- known-Z forcingを外したpersistent TVT-rateは、狙ったrate lagを改善せず、
  matched controlを大幅に壊した。
- 事前契約どおりrate定義/span/momentum/noise/grid/emission/prior/gate/
  blend/selectorを救済せず、branchをterminal closeする。
- rerun、Stage 1、inference、submissionは実行しない。

## 再現性メモ

- seed policy: RNGなし、固定well/row/position/TVT-rate/edge/message/reduction順。
- CPU/GPU: Kaggle private CPUで完了、GPU 0。
- SHA: input、rate grid/kernel、joint transition、posterior、prediction、
  diagnostic、metricsを実行時に保存する。
- gzipはdecompressed content SHAを主証拠にする。
- model / submission SHAは対象外。
- 初回runはdeterministic anchorとせず、独立rerunでSHA一致を確認する。

## 次のアクション

追加実行はない。exp446をnegative evidenceとして保持し、Stage 1、rerun、
inference、submission、same-fixed32救済を行わない。
