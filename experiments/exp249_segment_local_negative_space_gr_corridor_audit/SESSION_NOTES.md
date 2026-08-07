# exp249_segment_local_negative_space_gr_corridor_audit セッションノート

## 目的

exp246で失敗したfull-tail global hard-history barrierと、ユーザー着想の局所segment heatmapを分離する。固定candidateを変更せず、segment-local ridge/corridor eventのrisk濃縮と誤警報を監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle Stage 1 Version 2 COMPLETE、guard failed、不採用
- active mode: `stage1_full_audit`
- Stage 1: 773-well audit完了
- inference / submit: disabled

## 実装前コストガード

- active diagnostic mode: 1 (`stage0_preview`)
- LightGBM config: 0
- model training fold: 0
- booster: 0
- PF/Beam/likPF再生成: 0
- parent/control再学習: なし
- GPU: なし
- raw-test inference / submit: なし

## 固定した局所contract

- exp202: horizontal 128 rows、typewell 64 bins、target-free prior中心±192 ft。
- exp208: dense row-center stride 64、tail stopを含める。
- exp249: horizontal segmentとtypewell cropを別々にmedian/IQR scaleし、well全tail surface/normalizationを作らない。
- barrier/component/historyはsegmentごとにresetする。
- overlap判定はOR/AND/majority統合せず、agreementとinverse-coverage weighted readoutだけを作る。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-14に確認。
- seed policy: `no_new_rng_sorted_well_segment_local_audit`。seed 42はconfig contractのみ、新規RNGなし。
- upstream stochastic component: exp072固定PF/Beam candidate cacheのみ。candidate再生成なし。
- sorted well、single process、deterministic segment centers、global RNGなし。
- CPU、GPU/internet disabled、num_workers 1。
- candidate cacheはraw/decompressed SHA、raw filesはinventory SHA、gzip outputはdecompressed content SHAを主証拠にする。
- model / prediction / submissionは生成しない。本実験をdeterministic submission anchorとは扱わない。

## コマンドログ

### 2026-07-14 steering / experiment作成

```bash
make new-steering EXP=exp249_segment_local_negative_space_gr_corridor_audit
make new-exp EXP=exp249_segment_local_negative_space_gr_corridor_audit SOURCE=experiments/exp246_negative_space_gr_barrier_audit
```

- exp246のself-contained audit notebookを構成参照元としてコピーした。
- `.steering/20260714-exp249-segment-local-negative-space-gr-corridor-audit/`にStage 0/1、再現性、禁止事項を記録した。

### 2026-07-14 実装

- Stage 0 preview plot / pixel metadata / manifestを実装。
- Stage 1 segment path summary、flagged event、candidate/group/overlap/boundary/by-well metricsを実装。
- segment reset、ridge crossing、component transition、inverse overlap weightのsynthetic assertionsを実装。
- train notebookは1,891行・13 markdown cell、親exp246は1,454行・12 markdown cell。実験に必要な処理だけを持つself-contained構成とした。
- inference notebookはtrain-side diagnostic以外を拒否し、prediction / submissionを生成しないguardとした。

### 2026-07-14 静的・synthetic検証

```bash
.venv/bin/python -m py_compile experiments/exp249_segment_local_negative_space_gr_corridor_audit/*.py
.venv/bin/ruff check experiments/exp249_segment_local_negative_space_gr_corridor_audit/*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp249_segment_local_negative_space_gr_corridor_audit/exp249_segment_local_negative_space_gr_corridor_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp249_segment_local_negative_space_gr_corridor_audit/exp249_segment_local_negative_space_gr_corridor_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp249_segment_local_negative_space_gr_corridor_audit/*.py
make validate-exp EXP=exp249_segment_local_negative_space_gr_corridor_audit
```

- すべてpass。
- synthetic core contractsはpass。
- synthetic well readoutは2,238 view rows / 24 segment summariesを生成してpass。
- ローカルnotebook本実行は行っていない。Stage 0画像生成はKaggle CPU notebookを正とする。

### 2026-07-14 Kaggle package準備

```bash
make prepare-kaggle-notebooks EXP=exp249_segment_local_negative_space_gr_corridor_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp249-segment-local-gr-corridor-audit-train --title 'exp249 segment local gr corridor audit train' --run-on-push --strict"
```

- package作成とstrict validationはpass。push / runは未実施。
- metadata: CPU、GPU off、internet off、competition sourceあり、exp072 / exp115 kernel sourceあり。
- train source/package SHA256: `22763f41aa801d139f666ed66de685e57b5dabd3f480ddda5b1f2ea91812e8cf`。
- config source/package SHA256: `dffc7a52fc808e59e0dc8dcd44a8213c15caa02001f05a3b3984146e104510b3`。

## スコープ分離

- 本実験は、復元した初期contract（128 rows × 64 bins、±192 ft、stride 64、segment-local normalization/reset）だけを実装する。
- 後から追加されたMD 256 ft / 有向グラフ / shuffled-control契約は、進行中の別作業として変更しない。
- `KAGGLE_DIRECTION.md`の後発backlog行・詳細契約と、別途作成されたsteeringには触れない。

## 次のアクション

追加threshold / segment / stride探索、downstream feature化、raw-test inference、submitには進まない。後発の別契約は本実験の結果で変更せず、その実装作業を保持する。

### 2026-07-14 Kaggle Stage 0 Version 1実行

```bash
make push-kaggle-train EXP=exp249_segment_local_negative_space_gr_corridor_audit
kaggle kernels pull kentookumura/exp249-segment-local-gr-corridor-audit-train -p /tmp/kaggle-pull/exp249-segment-local-gr-corridor-audit-train -m
```

- push時刻: 2026-07-14 14:17 UTC（2026-07-14 23:17 JST）。
- kernel: `kentookumura/exp249-segment-local-gr-corridor-audit-train`。
- Kaggle kernel id_no: `127068942`。
- Version 1をCPU、GPU off、internet offで開始。
- 実行対象は1 diagnostic mode (`stage0_preview`) / 0 LightGBM config / 0 fold / 0 booster / parent-control再学習なし。
- push後のmetadata pullに成功し、同じcanonical kernel IDの存在とexp072 / exp115 kernel sourcesを確認。
- Stage 0完了とpreview出力を監視中。別slugへの再pushは行わない。

### 2026-07-14 Kaggle Stage 0 Version 1結果

- status: COMPLETE。
- runtime: 185.900秒。
- output: `kaggle/output/train_v1`へpreview確認に必要な小さい生成物だけ取得。
- 3 wells (`000d7d20`, `00bbac68`, `d07aed8f`) × first/middle/last、計9 PNG。
- 全PNGは2,250 × 1,050、manifest記載SHAと取得後SHAが一致。
- pixel metadataは128 rows × 64 bins、TVT range 384 ft、grid step 6.095703125 ftを確認。
- signed color range `[-4, 4]`、absolute color range `[0, 4]`、TVT右向き、horizontal row下向きを目視確認。
- segment / typewell crop別median-IQR normalizationをmetadataと画像で確認。
- last previewは`unique_horizontal_rows=65`だが、exp202/208と同じ`row_center + offsets`のwell端clipであり契約どおり。
- decision: pixel / axis / normalization parity pass。

### 2026-07-14 Stage 1 push前コストガード

- active diagnostic mode: 1 (`stage1_full_audit`)。
- LightGBM config: 0、fold: 0、booster: 0。
- parent/control再学習: なし。exp072固定candidate cacheを読むだけ。
- GPU: なし、internet: なし、raw-test inference / submit: なし。
- `manual_parity_confirmed=true`、`enabled_after_stage0_confirmation=true`へ変更。

### 2026-07-14 Kaggle Stage 1 Version 2実行

```bash
make prepare-kaggle-notebooks EXP=exp249_segment_local_negative_space_gr_corridor_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp249-segment-local-gr-corridor-audit-train --title 'exp249 segment local gr corridor audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp249_segment_local_negative_space_gr_corridor_audit
```

- push時刻: 2026-07-14 14:26 UTC（2026-07-14 23:26 JST）。
- same canonical kernelのVersion 2として`stage1_full_audit`を開始。
- source/package config SHA256: `c81bc1f7c785a3afd00d11a42e082f4f02769fb2ca05d202a435e1d4d03571b2`。
- CPU、GPU off、internet off。候補変更・推論・提出なし。
- 773-well train-side auditを監視中。別slugへの再pushは行わない。

### 2026-07-15 Kaggle Stage 1 Version 2結果

- status: COMPLETE。
- runtime: 3,673.976秒（61.2分）。
- processed: 773 wells / 3,783,989 rows。
- decision: `segment_local_audit_guard_failed`。
- guard結果:
  - bad-candidate precision lift 0.917349 < 1.5: fail。
  - good-candidate false-alert 0.540627 > 0.02: fail。
  - truth instantaneous false-alert 0.536992 > 0.001: fail。
  - overlap disagreement 0.138378 > 0.05: fail。
  - hidden-like truth false-alert 0.523356 > 0.002: fail。
  - worst-well truth false-alert 0.813880 > 0.05: fail（well `41fb0192`）。
  - boundary-core good false-alert delta -0.129466 <= 0.01: pass。
- family別bad precision liftは`beam_mean` 0.984600、`hyb` 0.973831、`likpf_mean` 0.986695、`pf_ancc` 0.989304、`sc_ens` 0.944813。全familyで1未満。
- 1000+ aggregate lift 0.926499、good false-alert 0.535755で不通過。
- hidden-like spatial / typewell-purged truth false-alertは0.523356 / 0.519132。
- by-well median truth false-alert 0.561330、q95 0.768034、max 0.813880。
- summary SHA256: `5f240979534ac428249867fce4a520b18e116cbc47162b5e5724cd977f8f466f`。
- summaryと小さいcandidate/group/overlap/boundary/by-well CSVだけを`kaggle/output/train_v2`へ取得。大きいsegment/event gzipは取得せず、Kaggle summary内のraw/decompressed SHAを記録した。
- 候補変更、推論、提出は未実施。追加gridは禁止事項どおり行わない。
- 後から追加されたMD 256 ft / DAG / shuffled-control backlogとそのsteeringは変更していない。
