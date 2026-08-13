# exp440_ambiguity_gated_predictive_prior_hmm セッションノート

## 目的

GRから複数のTVTが同程度に支持される行で、現在行のGR emissionをneutralizeし、
親transition後のpredictive priorを維持する介入を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: full 773-well OOF完了・`stage1_full_oof_failed_closed`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 実装承認: 2026-07-29のユーザー依頼「exp440を実装してください」
- 正規train Notebook採用 / Kaggle package / Stage 0 push/run:
  2026-07-29のユーザー依頼「実行してください」で承認済み
- Stage 1 full-well確認: 2026-07-30のユーザー依頼
  「念のためfull wellsに進んでください」で明示承認
- inference / submission: 未承認
- inference / submission: 無効
- CV: なし。fixed32はmechanism-only
- LB: まだなし

### 2026-07-30 full OOF完了

- Kaggle private CPU shard version 1を4本すべてCOMPLETE。
- shard 0: 193 wells / 946,128 rows / 5,603.734 sec / 1.571 GB
- shard 1: 193 wells / 946,017 rows / 8,695.083 sec / 1.499 GB
- shard 2: 193 wells / 946,112 rows / 8,162.896 sec / 1.509 GB
- shard 3: 194 wells / 945,732 rows / 5,549.296 sec / 1.527 GB
- strict merge:
  `kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-merge`
  version 1、COMPLETE。
- merge/readout: 292.181 sec / peak RSS 4.773 GB。
- 773 wells / 3,783,989 rows、candidate HMM 773 well-runs、
  saved parent control rerun 0、model / booster / PF / Beam / GPU 0。
- technical gateは全PASS。freeze前truth/fold/role readは0。
- candidate RMSE `12.992063`、parent exp209 `11.938287`、
  gain `-1.053776 ft`でFAIL。
- positive fold `1/5`。fold 3だけ`+0.228675 ft`改善。
- ambiguous rows 653,589、SSE reduction `-21.3117%`でFAIL。
- by-well delta p95 `+11.631749 ft`、worst `+45.003490 ft`でFAIL。
- raw observed / missing、高欠損、MD 1000+、hidden-like spatial /
  typewell-purgedはすべて悪化。
- decision:
  `close_without_blend_selector_continuous_gate_or_same_oof_rescue`。
- Stage 0 FAIL closedの解釈は維持し、Stage 1 rerun、inference、
  submission、same-OOF rescueを禁止してterminal close。
- 小さい評価artifactだけを
  `artifacts/kaggle_stage1_v1`へ選択取得し、行数とSHAを照合した。

実行package SHA:

- shard notebook:
  `d475787e...a9f59 / 915f2f9f...0209b / d9dd8a88...52317 /
  c16710e5...cbe9`
- shard kernel metadata:
  `d39801a4...8f9a / ff50bbdc...c2e1 / 2eeffd17...488 /
  154d2684...f0db`
- merge notebook:
  `e53bac63edc51e6ad2f9d8570ae1c05346e34e691f128be8aaf3bab099b248cc`
- merge kernel metadata:
  `6a0ebf62371f9f317d818397152814de92991d25b0fac28e4ae5adfec7a3cd80`
- scientific contract:
  `6fb3c6d849ad9a257088423a8a1c34b21f8e35e27cbb18ef83a6a8d534f98676`

結果記録後、local compact / canonical trainへStage 1 rerun禁止guardだけを
追加した。両方26 cells / 3,775 source lines、cell source SHA
`d78b6d0acf55382b82d89a7cc41864bc415c86cef58355c38db9a813933da8f1`
で一致する。実行済み科学ロジック、threshold、candidate、gateは変更していない。

### 2026-07-30 full wells実行承認

- Stage 0 FAIL closedの科学的解釈は維持する。
- scientific candidate: 1（parameter / threshold / lambda変更なし）
- candidate exact-HMM well-runs: 773
- reporting folds: 5
- saved parent control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- Stage 0実測のsingle-kernel投影は35,365.85秒で既存9時間hard guardを
  超えるため、suffix rowsのdeterministic LPTで4 CPU shardsへ分ける。
- shard wells: `193 / 193 / 193 / 194`
- shard suffix rows: `946,128 / 946,017 / 946,112 / 945,732`
- raw identity: 773 wells / 3,783,989 rows /
  `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- 各wellは1 shardだけで1回実行し、strict merge後にtruth/fold/
  hidden-like roleをattachする。
- inference / submissionは対象外。

### 2026-07-29 Stage 0実行承認

- scientific candidate: 1
- candidate exact-HMM well-runs: 32
- reporting folds: 5
- saved parent control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- target: Kaggle private CPU、internet disabled
- 正規train Notebook採用、package、push/runのみ承認済み。
- Stage 1、inference、submissionは対象外。

正規train Notebook採用:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  --output experiments/exp440_ambiguity_gated_predictive_prior_hmm/\
exp440_ambiguity_gated_predictive_prior_hmm_train.ipynb \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/\
exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train.py
```

- Kaggle version 1実行時はcompact / canonicalとも24 cells、
  2,481 source lines。実行時cell source SHA:
  `1c1cd394f21d7a6f4ce7107ea6e53a25491e4a92d7997e19ad9e9b8595d6a3c3`
- 結果記録後にrerun fail-closed guardだけを追加し、local compact /
  canonicalは24 cells、2,486 source lines、cell source SHA
  `b33347000c5d90667c848523c4d96bc10af8de518cf3f0aca7d95279bb871517`。
- inference Notebookは未変更。

Kaggle package:

```bash
make prepare-kaggle-notebooks \
  EXP=exp440_ambiguity_gated_predictive_prior_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-train \
  --title 'exp440 ambiguity gated predictive prior hmm train' \
  --run-on-push --strict --no-src"
```

- private: true
- CPU / TPU / internet: false / false / false
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp209-joint-exact-parity-train`
- bootstrap: 8 files。固定3 assetの埋込・展開後SHAをconfigと照合してPASS。
- package notebook SHA:
  `882134776ddb05d83e78298e097f9f691b3ba56302f88c8869d6ce52c0541d62`
- kernel metadata SHA:
  `d2ae0c552744af16210307a48b4ddcf7fa8b397ebbe5e7e93b53aeabc3bdda9f`
- 実行承認状態へ更新後の専用pytest `13 passed`、exp408/411/440関連
  pytest `39 passed`、Jupytext、py_compile、Ruff、strict experiment /
  template validationを再度PASS。

初回pushは、self-contained notebookに不要なrepository `src/`も埋め込み、
package notebookが1,317,934 bytesになった状態でKaggle `SaveKernel` HTTP 400を
返した。kernelは作成されなかった。slug/titleは49文字で、同じ長さの既存kernelが
正常なため、未使用`src/`を`--no-src`で除外した。科学コード、config、固定3 asset、
親kernel sourceは不変。914,607 bytesへ縮小し、bootstrap展開後SHAと実行契約を
再検証してPASSした。

### Kaggle Stage 0 version 1

- kernel:
  `kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-train`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp440-ambiguity-gated-predictive-prior-hmm-train`
- version: 1
- id_no: `129064462`
- started: `2026-07-29 14:07:11 UTC`
- completed: 約`2026-07-29 14:32:00 UTC`
- status: `KernelWorkerStatus.COMPLETE`
- scientific candidate 1、32 HMM well-runs、control再実行0、
  model/booster/PF/Beam/GPU各0。

### Stage 0 version 1結果

- 32/32 wells、156,088 suffix rows完走。
- elapsed: `1,464.045394482 sec`
- peak RSS: `1.058250427 GB`
- full 773-well runtime projection: `35,365.846560 sec`
- technical gate: `13 / 15 PASS`
  - FAIL: ambiguity activation `0.180526370 > 0.10`
  - FAIL: runtime projection `35,365.846560 > 30,600 sec`
- mechanism gate: `2 / 8 PASS`
  - predictive-better率:
    `0.421179644 < 0.55`
  - positive fold:
    `1 / 5 < 4 / 5`
  - ambiguous-row SSE reduction:
    `0.122000965 >= 0.05`、PASS
  - persistent-episode SSE reduction:
    `0.263619930 >= 0.05`、PASS
  - persistent改善:
    `7 / 16 wells < 10`、`3 / 5 folds < 4`
  - matched-control pooled RMSE delta:
    `+0.533491943 ft > +0.02 ft`
  - matched-control by-well delta p95:
    `+2.582386556 ft > +0.25 ft`
- finite coverage `1.0`、maximum normalization error
  `4.957747e-08`、no-ambiguity parent parity最大差`0.0 ft`、
  truth/role/fold/episode read before freeze `0`、全readback SHAはPASS。
- fixed32 pooled RMSEはparent `9.968802828`、candidate `9.280512860`、
  delta `-0.688289968 ft`。ただしpersistent 16 wellsへの偏りによる
  mechanism-only値で、control 16 wellsは`+0.533491943 ft`悪化しているため、
  CV・promotion evidenceとして扱わない。
- decision:
  `stage0_fail_closed_without_ambiguity_lambda_threshold_or_transition_rescue`
- Stage 1 eligibility: false。rerun、Stage 1、inference、submissionなし。

実ファイル監査のためoutputを
`/tmp/kaggle-output/exp440_ambiguity_gated_predictive_prior_hmm/train_v1`
へ取得し、行数、raw SHA、decompressed SHAをKaggle `metrics.json`と照合した。

- ambiguity schedule: 156,088 rows、decompressed
  `43502c0e...35b1`
- predictions: 156,088 rows、decompressed
  `f25e39b5...652`
- target-free diagnostics: 156,088 rows、decompressed
  `8fa194b8...6cc`
- ambiguous truth-late readout: 28,178 rows、
  `1eae525f...94c2`
- episode truth-late readout: 25 rows、
  `946c108c...8e2`
- well metrics: 32 rows、
  `81e90ebc...5485`
- input manifest:
  `83e4b7c5...5ffe`
- summary:
  `5375d693...d062`

## コマンドログ

### 2026-07-29 design-only作成

```bash
make new-steering EXP=exp440_ambiguity_gated_predictive_prior_hmm
make new-exp EXP=exp440_ambiguity_gated_predictive_prior_hmm
```

### 2026-07-29 compact self-contained実装

作成:

- `exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train.py`
- `exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train.ipynb`
- `exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_inference.py`
- `exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_inference.ipynb`
- `experiments/exp440_ambiguity_gated_predictive_prior_hmm/tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py`

検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/\
exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/\
exp440_ambiguity_gated_predictive_prior_hmm_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/*compact_selfcontained*.py \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py
.venv/bin/pytest -q experiments/exp440_ambiguity_gated_predictive_prior_hmm/tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py
.venv/bin/pytest -q \
  experiments/exp408_hmm_message_rate_basin_audit/tests/test_exp408_hmm_message_rate_basin_audit.py \
  experiments/exp411_predictive_filtered_rate_innovation_destick/tests/test_exp411_predictive_filtered_rate_innovation_destick.py \
  experiments/exp440_ambiguity_gated_predictive_prior_hmm/tests/test_exp440_ambiguity_gated_predictive_prior_hmm.py
make validate-exp EXP=exp440_ambiguity_gated_predictive_prior_hmm
make validate-template
```

結果:

- 専用pytest: `13 passed`
- exp408/411/440関連pytest: `39 passed`
- Jupytext `--test`: train / inferenceともPASS
- `py_compile`: PASS
- Ruff全選択ルール: PASS
- strict experiment validation: PASS
- template validation: PASS
- `task validate-exp`は`task: command not found`のため、同等の
  `make validate-exp`を使用した。
- 親compact比較:
  - exp408: 10章、2,483行
  - exp411: 9章、2,255行
  - exp440: 10章、2,547行
  - exp440はinput、exact HMM、gate、truth-late、metrics/orchestrationを
    notebook上で追えるため、親compactより薄いentrypointではない。
- `__file__`: compact train/inferenceとも0件。
- 正規train NotebookはStage 0実行承認後にcompact候補を採用。
- 正規inference Notebookは未変更。
- 全体`make test`は1,485件をcollect中、exp440と無関係な既存5実験の
  config-contract不整合でcollection停止した。
  - exp297: Stage-2 scientific contract mismatch
  - exp301: `execution.implementation_authorized`欠落
  - exp333: frozen Stage 0/1 contract mismatch
  - exp336 / exp349: experiment name contract mismatch
  - exp440のtest collection / assertion failureではないため、この依頼では
    他実験のconfigを変更しない。

## 変更点

- GMMは要件から外し、親exp209のGaussian emissionを維持した。
- candidateのcausal predictive messageへ通常emissionを一度適用した
  provisional filtered TVT marginalをexp236固定thresholdで判定する。
- raw GR observedかつambiguousの行だけemission exponentを0にし、
  predictive distributionをfiltered distributionとして維持する。
- 点推定TVTのhard freeze、soft lambda、threshold grid、GMM/Student-t/Huber、
  transition/prior/grid/sigma変更を禁止した。
- Stage 0をcandidate 1本 × fixed32 = 32 HMM well-runs、
  saved parent control再実行0に固定した。
- Stage 1をcandidate 1本 × 773 HMM well-runs、saved parent control再実行0とし、
  Stage 0全PASSと別承認を必須にした。
- model / LightGBM config / trained fold / booster / PF / Beam / GPUは
  Stage 0/1とも`0 / 0 / 0 / 0 / 0 / 0 / 0`。
- `docs/06_reproducibility.md`を読み、truth-late freezeとSHA契約を記録した。
- `backlog/KAGGLE_DIRECTION.md`ではP3とし、exp434 P1 / exp436由来fixed-five P3より後、
  P4原因分解より前に置く。

## 実装内容

- exp209と同じTVT/rate state、rate transition、position transition、
  Gaussian GR emission、prefix sigma、grid、prior、forward-backwardを
  self-containedで実装した。
- 各rowでcandidate predictive jointへ通常emissionを一度適用し、
  provisional filtered TVT marginalを作る。
- exp236と同じboundary peak、tie、top-2 ranking、valley split、
  mass定義と固定5 thresholdをNumba関数として実装した。
- gate disabled時のmean/std/log-likelihoodを親exp209正規
  `_hmm2_fb`と独立に照合し、`1e-10` absolute toleranceでPASSした。
- raw GR observedかつbimodalのrowだけcandidate filtered jointを
  predictive jointへ戻し、emission lambdaを`0.0`とする。
- forwardで完成したambiguity scheduleをbackwardのrow-wise lambdaとして
  そのまま再利用する。missing rowは親emission処理を維持する。
- outputは親と同じsmoothed posterior meanに加えてstdを保存する。
- fixed32 manifestはfreeze前に`well`列だけを読み、role/foldは
  全32 wellのschedule/prediction/diagnostic SHA freeze後に再読する。
- truth、persistent episode、exp408 causeも全freeze後にだけ読む。
- Stage 0のtechnical/mechanism gateは事前登録値のANDで判定し、
  1件でもFAILならno-rescueで閉じる。

## 科学的根拠

- exp408: current-row GRによるtruth優勢からwrong優勢への新規反転は
  `9 / 807,710 rows`。persistent episode SSEの`59.3978%`は
  current emission前のforward transition/prior hysteresisだった。
- exp236: 二峰rowは`35,399 / 3,783,989 = 0.9355%`。
  exp440はこの実験の固定thresholdをそのまま使う。
- exp133: broad ambiguity flagは`56.6857%`で、単純な高誤差gateではなかった。
- exp363: sticky weak-reliability signalはpooled AUC`0.607552`だったが、
  weak mass`0.589441`とhidden-like spatial gateをFAILした。

これらは本仮説を支持する証拠ではなく、Stage 0で早期反証する理由である。

## 実行量契約

- Stage 0:
  - scientific candidate: 1
  - candidate HMM well-runs: 32
  - parent control rerun: 0
  - reporting folds: 5
- Stage 1（Stage 0全PASS・別承認時のみ）:
  - scientific candidate: 1
  - candidate HMM well-runs: 773
  - parent control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 再現性メモ

- seed policy: RNGなし、well/row/position/rate/message順固定。
- stochastic components: なし。
- CPU/GPU runtime: CPU-only、GPU 0、internet disabled。
- Kaggle kernel: version 1 / id_no `129064462` / COMPLETE。
- input SHA: fixed32、exp209 saved control、exp408 episode/causeをconfig固定。
- ambiguity schedule / prediction / diagnostic SHA: 実行時にlogical/content SHAを記録。
- model manifest / model SHA: 非該当。
- submission SHA: 非該当。
- rerun check: 初回runはdeterministic anchorではない。Stage 0 FAILのため
  独立rerunを行わずbranchを閉じる。

## 次のアクション

1. `stage1_full_oof_failed_closed`として完了記録を維持する。
2. Stage 1 rerun、inference、submissionへ進まない。
3. threshold、lambda、GR scale、transition、prior、grid、blend、selector、
   well/row gateのsame-OOF rescueを行わない。
4. exp441--444のrate-transition別仮説はexp440救済として扱わず、
   それぞれの既存判断を維持する。
