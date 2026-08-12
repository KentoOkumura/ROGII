# exp444_acceleration_state_exact_hmm セッションノート

## 目的

exp441の全support OUへ3値acceleration状態を加え、rate変化の方向を
複数行に持続する明示trend-memoryがpersistent lagを回復できるか検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0A technical FAIL、runtime projectionによりterminal close
- 構造参照・一要因control: exp441、root比較対象: exp209
- 優先度: P4 high-risk
- 実装: 承認済み・完了
- 正規train Notebook採用 / package / Stage 0A: 承認済み
- Stage 0B/1 / inference / submission: 未承認
- CV / LB: なし

## 2026-07-30 先行条件判定

- exp441はStage 0でruntime projectionと5件のmechanism gateをFAIL。
- exp442は後続のユーザー判断でexp209直接比較の独立仮説へ変更された。
- exp444が要求する「exp441全gate安全かつlag残存、exp442不足/unsafe」は
  成立しなかった。

## 2026-07-30 独立仮説への変更

ユーザー指示:

```text
独立仮説としてください
```

- exp441/exp442の成否を実装・Stage 0Aの先行条件から外した。
- exp441のFAILはnegative contextに固定し、positive evidence、parameter/gate変更、
  same-fixed4 rescueには使わない。
- acceleration値、transition、initial prior、fixed4/fixed32、runtime/memory gateは
  当初契約から変更しない。
- fixed32では保存exp441を一要因readout、保存exp209を安全性controlにする。
- Stage 1はterminal close済みexp441を再実行せず、保存exp209比較を正とする。
- 元の「exp444を実装してください」を継続し、実装だけを承認済みとした。

## 2026-07-30 Stage 0A実行承認

ユーザー指示:

```text
実行してください
```

- compact trainを正規train Notebookへ採用した。
- Kaggle private CPU packageとfixed4 Stage 0Aを承認済みとした。
- scientific variant `1`、candidate HMM well-runs `4`。
- LightGBM config / trained fold / booster / fitted model / parent-control rerun /
  PF / Beam / GPUはすべて`0`。
- canonical kernel id/titleは
  `kentookumura/exp444-acceleration-state-exact-hmm-train` /
  `exp444 acceleration state exact hmm train`。
- Stage 0B/1、inference、submissionは引き続きfail-closed。

### Package / push

```bash
task prepare-kaggle-notebooks ...
# task command unavailable
make prepare-kaggle-notebooks \
  EXP=exp444_acceleration_state_exact_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp444-acceleration-state-exact-hmm-train \
  --title 'exp444 acceleration state exact hmm train' \
  --run-on-push --strict"
kaggle kernels push \
  -p experiments/exp444_acceleration_state_exact_hmm/kaggle/train
```

- `task` executableがないため、同等のMakefile経路でpackageを生成した。
- bootstrap configを展開し、canonical/package/Stage 0A、`run_hmm`、
  `create_prediction`だけがtrue、Stage 0B/1、inference、submissionがfalse、
  CPU、internet offであることを確認した。
- bootstrap fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- package notebook SHA:
  `764e40fa9cf36a326282fa794e29468c6cffdc5eeb561eca252d205a7514fae3`
- package metadata SHA:
  `6dca42e878f819802df96f00b1df5c6af014a0357f1bad147dfdef4a4debaf33`
- pre-push canonical pullは403で既存kernelを確認できなかった。push後のpullは成功。
- Kaggle version 1、id_no `129154702`、private CPU、GPU/internet off。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp444-acceleration-state-exact-hmm-train`

### Stage 0A version 1結果

- Kaggle status: `COMPLETE`
- 4 wells / 21,962 rowsを完走。
- candidate HMM `746.353694 sec`、Notebook内elapsed `767.339096 sec`。
- fixed32換算`5,970.829552 sec > 3,600 sec`: FAIL。
- full 773 wells換算`144,232.851372 sec > 30,600 sec`: FAIL。
- peak RSS `2.282776 GB <= 25 GB`: PASS。
- finite coverage `1.0`: PASS。
- acceleration row-sum誤差`1.110223e-16`: PASS。
- zero-acceleration exp441 OU parity誤差`0.0`: PASS。
- posterior normalization誤差`7.371881e-14`: PASS。
- dense reference prediction誤差`3.219611e-09`、
  posterior最大誤差`2.876623e-08`: PASS。
- truth/role/fold/episodeのfreeze前read `0`: PASS。
- 11 technical checks中、runtime projection 2件だけFAIL。
- Stage 0B eligible: `false`。
- fail action:
  `close_without_state_count_span_transition_kernel_runtime_or_memory_rescue`。

Kaggle logsにsummaryとSHAが十分出ているため、output archiveは取得していない。
主要decompressed SHA:

- prediction:
  `4927083191857ebf03dfd3ec755d2852afeb6125b4190e86796bc67552a2cfb1`
- acceleration posterior:
  `8d9f3b657e0904d79af7bfc07fa4b08ac71fbbfaed28338cf38ee2526a6498e3`
- target-free diagnostic:
  `b538e024c4f904fc210314929266b7be7b8cb73d375cbb01a2e4d30580d519d7`
- scientific contract:
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`

## コマンドログ

### 2026-07-29 design-only作成

```bash
make new-steering EXP=exp444_acceleration_state_exact_hmm
make new-exp EXP=exp444_acceleration_state_exact_hmm
```

## 設計契約

- stateを`(TVT, rate, acceleration)`へ1要因拡張する。
- acceleration値`[-0.0005,0,+0.0005]`、transition`0.08/0.84/0.08`。
- initial accelerationはzeroへ確率1。
- exp441 OU kernel、exp209 position/emission/prior/readoutは固定。
- Stage 0A 4 wells、PASS時に追加28でfixed32 total 32。
- Stage 1は全gate PASS・別承認時のみ773 wells。
- one-factor readoutは保存exp441、安全性/promotion controlは保存exp209。
- exp441/exp209 control rerunは0。
- parent rerun、ML model、booster、PF、Beam、GPUは0。
- state数、span、transition、prior、emission、gate救済は禁止。

## 再現性メモ

- RNGなし。Stage 0Aはwell identityだけの固定SHA順。
- acceleration/joint kernel、posterior、prediction、diagnostic SHAを保存する。
- exp441 fixed32/full prediction SHAは各stage前に厳密固定する。
- truth/role/fold/episodeはfreeze後だけjoinする。
- 初回runはdeterministic anchorとしない。

## 2026-07-30 Stage 0A実装

作成:

- `exp444_acceleration_state_exact_hmm_compact_selfcontained_train.py`
- `exp444_acceleration_state_exact_hmm_compact_selfcontained_train.ipynb`
- `exp444_acceleration_state_exact_hmm_compact_selfcontained_inference.py`
- `exp444_acceleration_state_exact_hmm_compact_selfcontained_inference.ipynb`
- `experiments/exp444_acceleration_state_exact_hmm/tests/test_exp444_acceleration_state_exact_hmm.py`

実装内容:

- 3状態acceleration transitionを
  `[[0.92,0.08,0],[0.08,0.84,0.08],[0,0.08,0.92]]`に固定し、
  initial probabilityを`[0,1,0]`に固定した。
- destination accelerationでOU meanを`a_t*delta_MD`だけ動かす
  full-support bin-integrated rate kernelを実装した。
- joint stateを密な巨大行列へ展開せず、`acceleration -> rate -> position`の順に
  因子化したexact forward/backwardを実装した。position後にcurrent GR emissionを
  一度だけ適用する。
- zero acceleration sliceとexp441 OU kernelの数値parity、acceleration row sum、
  small-state dense forward/backwardとのposition/acceleration posterior parityを
  contract化した。
- fixed32 manifestは`usecols=["well"]`だけで開き、
  `SHA256("exp444_runtime_preflight" + well)`順の先頭4 wellsを選ぶ。
- Stage 0Aはtruth/role/fold/episode/causeを一切開かず、prediction、
  acceleration posterior、rate diagnostic、joint transition SHAをfreezeする。
- 4-well実時間からfixed32/full runtimeを投影し、`3,600 / 30,600 sec`、
  RSS`25 GB`をAND gateにする。
- inference候補は実装承認だけを許し、Stage 0A/0B/1、inference、
  submissionをfail-closedにした。

実行契約:

- scientific variant: `1`
- Stage 0A / Stage 0B total / Stage 1 candidate HMM well-runs:
  `4 / 32 / 773`
- parent/control HMM rerun: `0`
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- ローカルNotebook実行: `0`
- Kaggle package / Stage 0A version 1: 後続の明示承認で実行済み

親compact比較:

- exp441 / exp444 train source: `3,070 / 2,537 lines`
- 両方とも10章構成。
- exp444はinput preparation、acceleration/OU kernel、exact forward/backward、
  dense reference、identity-only fixed4、target-free SHA freeze、gate、guarded
  orchestrationをNotebook上で追える。
- 同一exp helper import、`__file__`、薄い`main()` entrypointはない。
- compact train / inference: `24 / 8 cells`

検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp444_acceleration_state_exact_hmm/*compact_selfcontained*.py \
  experiments/exp444_acceleration_state_exact_hmm/tests/test_exp444_acceleration_state_exact_hmm.py
.venv/bin/ruff check \
  experiments/exp444_acceleration_state_exact_hmm/*compact_selfcontained*.py \
  experiments/exp444_acceleration_state_exact_hmm/tests/test_exp444_acceleration_state_exact_hmm.py
.venv/bin/pytest -q experiments/exp444_acceleration_state_exact_hmm/tests/test_exp444_acceleration_state_exact_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp444_acceleration_state_exact_hmm/\
exp444_acceleration_state_exact_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp444_acceleration_state_exact_hmm/\
exp444_acceleration_state_exact_hmm_compact_selfcontained_inference.py
make validate-exp EXP=exp444_acceleration_state_exact_hmm
make validate-template
```

- 専用pytest: `14 passed`
- py_compile: PASS
- Ruff全選択ルール: PASS
- Jupytext train / inference round-trip: PASS
- strict experiment validation: PASS
- template validation: PASS
- reviewer: Core evidence categories present
- forward normalizationはpredictive/filtered全状態の正規化済み確率を
  独立に再加算する実監査へ最終修正した。

SHA:

- compact train source:
  `fed964e1edad3aba1d9b2347c45e22476cfd194fcf1172cb6e0f0dd094eb3041`
- compact train ipynb:
  `eb505f2c9708833b53483d5651800910dbb7c7f35767893dd7f1f5ac32f67794`
- compact inference source:
  `cd9b26c39f636203a68bbb14907c93d2e38c0e5bd94fdd213cf1459c5bfa5402`
- compact inference ipynb:
  `bc0a090393b838e67f93fd00c0bd34768c0f92dff22002babc63dcdbf2edc37f`
- 専用test:
  `1c8ae3d2b53a4e7bdc364dfe9409ee6559242e589a35642b4f815aa259f1afc5`
- 採用済み正規train Notebook:
  `cc66b05b7a16071a94fc2d6b1e99d6e417ca655d6d7bcbec97abb3b8eef7724f`
- 未変更の正規inference scaffold:
  `4506c9c45bc901dc8ffd049d5e1790d75101a209b23ef3dc754324b8d62cd636`
- 実行済みpackage config:
  `cefc49100aa3a1c9018e87ea84514cabdf0bd2fe2d4d6f519d5d579b1ff96b57`
- terminal-close反映後のlocal config:
  `a8a6492e3d4a43e876f0440186c3511d7210cd16aabcf9d9252164d8911bc0e2`

## 次のアクション

1. exp444をterminal closeし、Stage 0B/1へ進まない。
2. state数、span、transition、kernel、runtime実装、parameter、gateを
   exp444内で救済しない。
3. inference、submissionは無効のままにする。
