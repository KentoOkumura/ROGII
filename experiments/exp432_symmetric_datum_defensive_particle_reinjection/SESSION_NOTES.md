# exp432_symmetric_datum_defensive_particle_reinjection セッションノート

## 目的

方向を使わない±datum defensive proposalで、PFの有限粒子supportを一度だけ
回復させる高リスク案を実装し、Stage 0の実行前contractを固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU Stage 0 version 1完了、`stage0_fail_closed`
- CV / LB: なし
- 実装承認: あり（ユーザー指示 `exp432を実装してください`）
- 正規Notebook採用 / Stage 0 package・push・run: 完了
- full / inference / submission: 不実施、再実行も再ロック

## 2026-07-28 設計記録

- exp410のfinite support原因証拠と、exp412のtarget-free rate-gap event時刻を系譜化した。
- exp412のbeta方向は不使用。negative resultは再分類しない。
- proposalをbase/minus/plus=`0.80/0.10/0.10`、datum floor=`0.35 ft`、最大1 eventに固定した。
- importance ratioはfull mixtureの`p0/q`、clipなし、理論上限1.25。
- Stage 0はHMM 32、baseline PF 32、treatment PF 32、LightGBM config/fold/booster=`0/0/0`。
- full予定はHMM trigger 773、treatment PF 773、親PF独立full rerun 0。
- 設計時点では実装コード、実行 notebook、Kaggle kernelを作成していなかった。

## 2026-07-28 実装記録

- `kaggle-review-exp`と`docs/06_reproducibility.md`を確認した。
- ユーザー指示`exp432を実装してください`を、設計済みStage 0のcompact
  self-contained source / 候補Notebook / contract test実装承認として扱った。
  正規Notebook採用、Kaggle package/push/run、full、inference、submissionは
  既存どおり別承認とした。
- Jupytext percent source
  `exp432_symmetric_datum_defensive_particle_reinjection_compact_selfcontained_train.py`
  を実装し、同名のcompact候補Notebookへ変換した。
- notebookには次をセル上で追える形で展開した。
  - fixed32 scopeだけを先に読むtruth-late leakage ledger
  - exact exp209 first-pass HMM、filtered/smoothed rate、filtered position std
  - exp412 inclusive 16-row persistent rate-gapと最初のfalse→true event
  - exact exp404 x1.0 PF baselineとcommon-random treatment
  - immutable experiment/well/seed/event/particle由来の独立component stream
  - event transitionだけのbase/minus/plus=`0.80/0.10/0.10`
  - full three-component densityに対するunclipped importance correction
  - particle support、branch ancestry、particle state、prediction、SHA ledger
  - 全target-free生成物freeze後のtruth / cause role / fold join
  - 512-row support/SSE、fold、matched controlのfail-closed AND gate
- `datum >= 0.35 ft`に対しposition noiseが`0.005 ft`なので、shifted branchの
  `p0/q`通常比はfloat64で0へunderflowし得る。scientific targetは変えず、
  finiteな`log p0 - log q`をlog-weightへ直接加え、log-sum-expで正規化する
  実装にした。ratio clip/floorは0。

### 実装時の固定実行量

- active scientific variants: `1`
- HMM trigger well-runs: `32`
- baseline / treatment PF well-runs: `32 / 32`、合計`64`
- seeds / particles: `128 / 500`
- seed-well trajectories / particle starts: `8,192 / 4,096,000`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- Beam / GPU: `0 / 0`
- full予定: HMM `773`、treatment PF `773`、親full PF再実行`0`

これは実装済みconfigの固定値であり、Kaggle実行承認ではない。

## 2026-07-29 Stage 0実行承認

- ユーザー指示`実行してください`を、compact候補の正規train Notebook採用と
  fixed32 Kaggle Stage 0 package / push / runの承認として記録した。
- 実行対象はactive scientific variant `1`、HMM `32` well-runs、
  baseline / treatment PF `32 / 32`、合計PF `64` well-runs、
  seed-well trajectories `8,192`、particle starts `4,096,000`。
- LightGBM config / fold / booster、Beam、GPUはすべて`0`。
- baseline PF 32 well-runsはno-event/common-random/control safety監査に必要な
  親control再実行であり、上記承認に含めた。
- full `773` wells、inference、submissionは解錠していない。

### push前package監査

- 初回canonical候補
  `exp432-symmetric-datum-defensive-particle-reinjection-train`
  （slug / titleとも59文字）は、Kaggle `SaveKernel` の詳細なし400で未作成だった。
- `kaggle kernels pull`は403、`kaggle kernels list --mine --search exp432`は
  `Not found`で、失敗したslugにkernelが作成されていないことを確認した。
- 原因はrepo内の既知事例と同じKaggle kernel slug/titleの50文字上限。
- 科学設定を変えず、44文字のcanonical ID/title
  `kentookumura/exp432-symmetric-defensive-reinjection-train` /
  `exp432 symmetric defensive reinjection train`へ同時に短縮して再packageする。
- 正規train Notebook SHA256:
  `ea8f9fd362c885a320405a992df3036c8769c04670f0b8b5bf3ce19b10528c40`
- Kaggle package Notebook SHA256:
  `5ff50dde48505e53cd3657d395577596beab4037c00ca570de623c158da04eca`
- package / repository `config.yaml`: byte一致
- private / CPU / GPU・TPU無効 / internet無効 / run-on-push: 確認済み
- input: 公式competition、exp209 kernel source、exp404 frozen predictions dataset

### Kaggle Stage 0 version 1

- push: 成功
- URL:
  `https://www.kaggle.com/code/kentookumura/exp432-symmetric-defensive-reinjection-train`
- kernel id_no: `128974856`
- pull後metadata: canonical id/title、private、CPU、internet無効、入力3系統一致
- docker image:
  `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`
- 初回確認status: `RUNNING`
- 実行中のKaggle CLI logs: 空。既知仕様どおり、空ログだけでは失敗判定しない。
- 最終status: `COMPLETE`
- Stage 0 notebook elapsed: `3,798.063204687秒`（約63.30分）
- peak RSS: `1.2447776794433594 GB`
- 送信packageとKaggle pull後Notebookは21 cells、連結source SHA256
  `07dfda43af46b43e4bfa344e9365bfa427320a966a6028ce9ea525d792bafcf3`
  で一致した。

### Stage 0結果

- status: `stage0_fail_closed`
- full eligible: `false`
- triggered / no-event wells: `21 / 11`
- HMM保存parent / PF保存parent / event前seed prediction max abs diff:
  `0 / 0 / 0 ft`
- no-event particle/prediction/support bitwise parity: PASS
- truth / role reads before all freeze: `0 / 0`
- finite log importance、quadrature、`max log(p0/q)=log(1.25)`: PASS
- technical gateは実行ログ上`13 / 14` PASS。唯一のFAILはfull runtime projection。
  NotebookはStage 0全elapsedを単一kernelへ`773/32`倍し、
  `91,746.96428822033秒 > 32,400秒`と判定した。
- 設計はHMM cacheと4 PF shardsを別実行にするため、全elapsedを4分割した
  保守的参考値は`22,936.741072055083秒`（約6.37時間）。実行Notebookの
  runtime gateが`cpu_pf_shards: 4`を反映していない注意点を残す。
- mechanism gate:
  - triggered wells `21 >= 8`: PASS
  - truth-outside-support率 baseline / treatment:
    `0.1883370536 / 0.1926153274`
  - support absolute reduction:
    `-0.0042782738 < +0.05`: FAIL（0.428 points悪化）
  - triggered-window SSE reduction:
    `0.1208714033 >= 0.10`: PASS
  - nonworse folds: `3 / 5 < 4 / 5`: FAIL
  - control pooled RMSE delta:
    `-1.0056944148 ft <= +0.02 ft`: PASS
  - control worst-well RMSE delta:
    `+0.5830729757 ft > +0.25 ft`: FAIL
- runtime projectionの実装上の注意点にかかわらず、mechanism gateが独立に
  3項目FAILしたためfull不適格の結論は確定する。fixed32上のgate救済、
  full、inference、submissionへ進まない。
- Kaggle output archiveは取得していない。完了logsにmetrics、fold、
  runtime/RSS、入力/生成物SHAが揃っているため、AGENTS.mdの既定に従った。

### 検証

- exp432専用test: `12 passed`
- exact exp209 first-pass HMM independent parity: PASS
- no-event exp404 seed prediction、log-likelihood、resampling count、minimum ESS、
  clip count bitwise parity: PASS
- event前baseline/treatment seed prediction/support bitwise parity: PASS
- stable component streamの再現性とbase/minus/plus mass:
  `0.80/0.10/0.10 ±0.01` synthetic PASS
- finite `log p0 - log q`、上限`log(1.25)`、importance quadratureでparent mass /
  mean / variance復元: PASS
- Jupytext round-trip、`py_compile`、Ruff F821: PASS
- parent compact比較:
  exp412 `2,333`行に対しexp432 `2,366`行。9章でHMM trigger、
  PF proposal、freeze、truth-late gate、生成物を追える。
- `Path(__file__)`、同一exp helper import: `0`
- 実装完了時点の正規Notebook、Kaggle package、push、run: `0`
  （2026-07-29に採用・実行済み）

## 再現性メモ

- HMM trigger: RNGなし、first-pass scheduleをtruth前にfreeze
- PF seed: immutable well id × seed index
- component seed: experiment × well × seed × event row × particle
- base/component RNG streamを分離し、shard/再開順に依存させない
- Kaggle実行で保存・記録済み:
  trigger/event/common-random/ancestry/importance/trajectory/prediction/metrics SHA
- deterministic anchor: stochastic rerun parity未確認のためfalse

## 次のアクション

1. exp432を`stage0_fail_closed`で閉じる。
2. fixed32上のmixture/datum/trigger/gate救済、full、inference、submissionは行わない。
3. support外率を悪化させたため、このproposalを後続案のpositive evidenceにしない。
