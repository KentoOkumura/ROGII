# exp426_rsd_binned_pattern_absolute_reanchor セッションノート

## 目的

Wu et al. (2019) のRSD-binned GR pattern scoreが、
HMM / exp226に不足するcoarse absolute datumを観測できるかを先に検証し、
通過した場合だけexp226再アンカーとPF absolute-anchor proposalへ進む。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage A technical FAIL・terminal close
- 優先度: P3・完了
- CV / LB: scientific評価未実施 / なし
- Stage A実装承認: あり・実装済み
- 正規Notebook編集: compact self-contained Stage Aを採用済み
- Kaggle package / push / run: version 1完了、id_no `128930757`
- inference / submission: 無効

## 2026-07-28 設計セッション

### 実行済みコマンド

```bash
make new-steering EXP=exp425_rsd_binned_pattern_absolute_reanchor
make new-exp EXP=exp425_rsd_binned_pattern_absolute_reanchor
```

scaffold作成中に別の`exp425_symmetric_datum_reanchor_exact_hmm`が追加されたため、
既存側を変更せず、この実験のsteering / experiment directory / internal nameを
`exp426_rsd_binned_pattern_absolute_reanchor`へ繰り上げた。

設計文書更新後に実行した確認:

```bash
make update-summary
make validate-exp EXP=exp426_rsd_binned_pattern_absolute_reanchor
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py \
  exp426_rsd_binned_pattern_absolute_reanchor --root .
```

- `make validate-exp`: strict PASS
- YAML / JSON parse: PASS
- experiment docs review: core evidence categories present
- `experiment_summary.md`: exp425 / exp426 / exp427の一意なlineageを確認

### 確定した設計

- Stage A:
  512-row block、固定13 offsets、RSD 0.5 ft bin mean、Pearson /
  signed Fisher scoreでabsolute offset identifiabilityを全773 OOF wellsで検証する。
- matched controls:
  raw pointwise Pearson、exp280互換raw Gaussian、stable score-label permutation。
- Stage B:
  scoreをcoarse datum Viterbiへ入れ、exp226 final pathへ線形補間した
  correctionだけを加える。
- Stage C:
  exp404 x1.0 / temperature-5 PFに10% uniform absolute-anchor targetを加え、
  同じtargetに対するuniform proposalとpattern-guided defensive proposalを比較する。
- guided proposal:
  original continuation 90%、uniform 13 anchors 5%、equal top-3 anchors 5%。
  `p_aug/q<=2`、importance clipなし、raw GR emission変更なし。
- fail-close:
  Stage A FAILならB/C、Stage B FAILならCを実装しない。

### 実行量

- Stage A/B:
  model / LightGBM config / trained fold / booster / HMM / PF / Beam / GPU =
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`
- Stage C0:
  2 PF variants ×12 sentinel wells = 24 PF well-runs、
  3,072 seed-well trajectories、1,536,000 particle starts。
- Stage C1:
  1 PF variant ×773 wells、98,944 seed-well trajectories、
  49,472,000 particle starts、4 CPU shards。
- parent control再実行: なし。

### 設計上の解釈

- exp408でcurrent pointwise GR emissionはroot causeでなかったため、
  HMM emissionの全面置換は行わない。
- Stage AはHMM residual-datum stateの共通必要条件だけを検証する。
- Stage Bがexp226への直接貢献、Stage CがPF物理モデルへの直接貢献を検証する。
- Stage Cではscoreをproposalにだけ使い、importance correctionでGR二重計上を避ける。

## 再現性メモ

- `docs/06_reproducibility.md`: 確認済み。
- Stage A/B seed: RNGなし。
- Stage C seed:
  well / split / variant / seed index由来のstable SHA256 local RNG。
- stochastic components:
  Stage C particle propagation、mixture draw、systematic resampling、roughening。
- CPU/GPU runtime:
  Kaggle private CPU、GPU / internetなし、well内single worker。
- SHA:
  input、config、code、score/rank/top-3、prediction、PF diagnosticsの
  logical / decompressed content SHAを記録予定。
- model / submission SHA:
  modelとsubmissionを作らないため非該当。
- deterministic anchor:
  false。Stage A/B independent rerun、Stage C fixed probe parityと
  raw-test regenerationが揃うまで昇格しない。

## 未実施

- compact self-contained source / tests
- 正規train / inference Notebook編集
- validation、Kaggle package / push / run
- Stage A / B / C
- inference / submission

## 次のアクション

別承認後にStage Aだけを実装する。結果を見てbin / block / offset / score familyを
救済せず、全gate PASS時だけStage Bの実装承認を得る。

## 2026-07-28 Stage A実装セッション

### ユーザー承認と境界

ユーザーの「exp426を実装してください」をStage A実装の明示承認として扱った。
正規train / inference Notebookの上書き、Kaggle package / push / run、
Stage B / C、inference、submissionは承認範囲外として実施していない。

### 実装

- compact self-contained Jupytext source:
  `exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py`
- 対応するnotebook候補:
  `exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.ipynb`
- contract tests:
  `tests/test_exp426_rsd_binned_pattern_absolute_reanchor.py`
- config:
  `stage_a_implementation_ready`へ更新。Stage A実装だけを有効化し、
  package / push / run flagはすべてfalseを維持した。

Stage Aはexp226保存OOFのfinal `tvt_pred`をbase pathとし、512-row block /
固定13 offsetsで次を同時計算する。

1. RSD 0.5 ft bin mean＋signed Fisher-Pearson primary
2. raw pointwise Pearson
3. exp280互換raw Gaussian
4. stable SHA256 score-label permutation
5. descriptive cosine / Spearman

score、support、rank、top-3、input / well manifest、logical SHA、
fixed probe rerunをfreezeした後だけ`tvt_true`とhidden-like roleを読む。
technical FAIL時はtruthを読まず、target-free生成物とfail-close summaryだけを保存する。
unsupported blockはoffset 0へfallbackし、replay RMSEはscope全row、
identifiability指標はprimary RSD supported blockだけで集計する。primary /
control比較も同じprimary support集合を分母とし、control固有invalidはoffset 0
fallbackとして扱う。

### 実行量

- active Stage A primary score: 1
- descriptive scores: 2
- matched controls: 3
- reporting folds: 5
- model / LightGBM config / trained fold / booster / HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`
- parent / control再学習: なし

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp426_rsd_binned_pattern_absolute_reanchor/\
exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py \
  tests/test_exp426_rsd_binned_pattern_absolute_reanchor.py
.venv/bin/ruff check \
  experiments/exp426_rsd_binned_pattern_absolute_reanchor/\
exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py \
  tests/test_exp426_rsd_binned_pattern_absolute_reanchor.py
.venv/bin/pytest -q tests/test_exp426_rsd_binned_pattern_absolute_reanchor.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp426_rsd_binned_pattern_absolute_reanchor/\
exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py
```

- contract tests: 8 passed
- syntax / ruff / ruff format: PASS
- 正規Notebook: untouched placeholder
- Kaggle run / CV / LB: 未実行 / なし

全リポジトリの`make test`も実行したが、exp426 testsを実行する前のcollectionで、
既存のexp297 / exp301 / exp333 / exp336 / exp349 sourceが各experiment configを
解決できず5 errorsで停止した。exp426対象test 8件、strict validation、
Jupytext parity、ruff、py_compileは独立にPASSしている。今回の実装と無関係な
既存実験は変更していない。

### 次のアクション

別承認後にcompact候補の章立てと内容を確認して正規train Notebookへ採用し、
Kaggle private CPU Stage Aのpackage / push / run前にbootstrap内config /
source SHAとinput sourceを照合する。Stage A全gate PASS時も、Stage Bは
別承認を得るまで実装しない。

## 2026-07-28 Stage A Kaggle実行セッション

### ユーザー承認

ユーザーの「実行してください」により、Stage A正規train Notebook採用、
Kaggle package / push / runを承認済みとした。Stage B / C、inference、
submissionは引き続き未承認。

### push前実行量

- active Stage A primary score: 1
- descriptive scores: 2
- matched controls: 3
- reporting folds: 5
- model / LightGBM config / trained fold / booster / HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`
- parent / control再学習: なし
- runtime: Kaggle private CPU、internetなし、single worker
- fixed input:
  exp226保存OOF、raw competition train、exp115 hidden-like assignment

### canonical package

- kernel id:
  `kentookumura/exp426-rsd-binned-pattern-absolute-reanchor-train`
- title:
  `exp426 rsd binned pattern absolute reanchor train`
- 正規train Notebook:
  compact self-contained Stage A候補を採用
- 正規inference Notebook:
  placeholderを維持

### Kaggle実行結果

- kernel:
  `kentookumura/exp426-rsd-binned-pattern-absolute-reanchor-train`
- version / id_no:
  `1 / 128930757`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp426-rsd-binned-pattern-absolute-reanchor-train`
- inventory:
  `3,783,989 rows / 773 wells / 5 folds / 7,787 blocks`
- score bank:
  `101,231 rows = 7,787 blocks × 13 offsets`
- target-free freeze runtime / total runtime:
  `160.789156 / 164.719113 sec`
- peak RSS before truth:
  `0.803265 GB`

Stage A technical gateは次の2条件だけがFAILした。

- supported block fraction:
  `0.255939386 < 0.95`
- supported well fraction:
  `0.892626132 < 0.98`

canonical order、duplicate identity zero、expected rows / wells / folds、
finite score、fixed offset order、13 offsets per block、rank permutation、
top-3 mask、runtime、memory、fixed-probe logical SHA parityはPASSした。
fixed probe wellは`000d7d20`で、期待値とrerunのlogical SHAはいずれも
`b4fd1730c8c2c53a07a0e326bc762b52f1c4249b226e2546f30544b215069c59`。

fail-closeによりtruthとhidden-like roleはfreeze前後とも読まず、read countは
`truth 0 / 0、hidden-like role 0 / 0`。scientific gate、CV readout、
Stage B / C、inference、submissionは実行していない。

### 再現性記録

- scientific contract SHA:
  `2fc2b35c9ea88aa3b9c35546e4979dbb335d9944a225c601b4810711d6c164ca`
- config content SHA:
  `2d3bf3a7955342c5f067fd83e93188c0a5e9cf12f0931de636de7aded5f861fa`
- score content SHA:
  `463aa32bef9a1045469466e2cf5fd68e038258e75f11fc88153fd9ca7f8dd2fd`
- score schema SHA:
  `6e86f76bf7df038e5b3b8077db80888c737bfa3880377ac24fe74d236706e9bd`
- input manifest content SHA:
  `7933f0f2babaa382ee23ae64db096db0dcc775035fc399254e64e7b30fe7656b`
- well manifest content SHA:
  `6af41b07945527af423405a860bd04ef356656266cfb264b80c471a72e7d0266`

train-side CVや後段入力として実ファイルを使う必要がなく、logs / cell outputで
gate、read ledger、SHAを確認できたため、Kaggle output archiveは取得していない。

### 終端判断

support coverageのtechnical FAILであり、offset識別精度の科学的negativeとは
判定しない。一方、事前登録したfail-closeに従い、同じOOFでbin幅、block、
offset、minimum points / paired bins、Type Well extrapolation、score familyを
変更する救済は行わない。exp426はterminal closeとし、Stage B / C、
inference、submission、同じRSD-binned score familyの後続backlogには進まない。

### 実行後検証

- exp426 contract tests: `8 passed`
- `ruff check` / `ruff format --check`: PASS
- `py_compile`: PASS
- Jupytext `--to ipynb --test`: PASS
- `make validate-exp EXP=exp426_rsd_binned_pattern_absolute_reanchor`:
  strict PASS
- YAML / JSON parse: PASS
- experiment docs review: core evidence categories present
- `make update-summary`: final statusを反映済み
