# exp246_negative_space_gr_barrier_audit セッションノート

## 目的

horizontal×typewell GR mismatchの赤いridgeを、正しいpath生成ではなくnegative-space edge barrierとして利用できるか監査する。初回は固定candidateを一切変更せず、true-path survivalとcandidate exclusion precisionを測る。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU train v2完了、5 safety guardsすべてfail、hard barrier不採用
- CV / LB: なし
- inference / submit: disabled

## 実装前コストガード

- active variant: 1 (`diagnostic_only`)
- LightGBM config: 0
- model training fold: 0
- booster: 0
- PF/Beam/likPF再生成: 0
- 親/control再学習: なし
- GPU: なし
- hidden-test inference / submit: なし

## 変更点

- exp202/215のheatmap signalとpath-ranking失敗を踏まえ、path生成ではなくexclusionだけを実装。
- exp072固定candidate cacheを`usecols` + 200,000-row chunkでwell streamする。
- raw/smoothed GR差の固定threshold、5-row持続、12ft厚み、state cap 256でbarrierを作る。
- missing/flat/全面赤rowをunsupported neutralにする。
- anchor-connected corridor、candidate endpoint/ridge crossing、history後validを保存する。
- true-path safety guardはendpoint forbiddenだけでなく、anchor corridor外とedge crossingを含む瞬時違反率で判定する。
- strict survivor 0件と未変更union fallbackを別指標にする。
- exp115 hidden-like role maskは評価groupにだけ使用する。

## 再現性メモ

- `docs/06_reproducibility.md`を2026-07-14に確認。
- seed policy: `no_new_rng_deterministic_barrier_audit`。seed 42はconfig contractのみで新規RNGなし。
- stochastic components: upstream exp072 PF/Beam candidate cacheのみ。
- well処理: sorted順、single process、global RNGなし。
- CPU/GPU runtime: CPU、GPU disabled、internet disabled、num_workers 1。
- input evidence: exp072 gzip raw SHA / decompressed content SHA、exp115 assignment SHA、raw file inventory SHA、config SHAをsummaryへ保存する。
- output evidence: row audit gzipはraw SHAとdecompressed content SHA、他CSVはfile SHAを保存する。
- model / prediction / submission SHA: 対象外。モデル・予測・submissionを生成しない。
- deterministic anchor: false。固定inputに対するaudit自体はdeterministicだがupstream候補生成のprovenanceを引き継ぐ。

## コマンドログ

### 2026-07-14 バックログ・steering・実験作成

```bash
make new-steering EXP=exp246_negative_space_gr_barrier_audit
make new-exp EXP=exp246_negative_space_gr_barrier_audit
```

- `KAGGLE_DIRECTION.md`へ未着手backlogを追加後、`docs/legacy/steering/20260714-exp246-negative-space-gr-barrier-audit/`へ切り出した。
- 新規templateを作成し、route / lineage / leakage / barrier / guards / reproducibilityをconfigへ固定した。

### 2026-07-14 実装・静的検証

```bash
.venv/bin/python -m py_compile experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_train.py experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_inference.py
.venv/bin/ruff check experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_train.py experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp246_negative_space_gr_barrier_audit/exp246_negative_space_gr_barrier_audit_inference.py
make validate-exp EXP=exp246_negative_space_gr_barrier_audit
make validate-template
make prepare-kaggle-notebooks EXP=exp246_negative_space_gr_barrier_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp246-negative-space-gr-barrier-audit-train --title 'exp246 negative space gr barrier audit train' --run-on-push --strict"
```

- py_compile: pass。
- ruff default rule set: pass。
- Jupytext convert/test: train / inferenceともpass。
- strict experiment validation: pass。
- project template validation: pass。
- strict Kaggle train package preparation: pass。
- kernel metadata: CPU / internet off / run-on-push、kernel sourceはexp072・exp115の2本、dataset sourceなし。
- notebook sourceに`__file__`なし。
- train notebookは10章で、薄いhelper呼出ではなく入力、barrier、corridor、candidate監査、metrics/SHA保存を展開した。
- 初回notebook実行はKaggleを正とするため、ローカルsmoke/full executionは行っていない。

## 実行前予定（完了）

```bash
make push-kaggle-train EXP=exp246_negative_space_gr_barrier_audit
```

push前にactive variant 1、config/fold/booster `0/0/0`、parent/control再学習なし、CPU、internet disabled、2 kernel sourcesをpackage/bootstrapで再確認する。

### 2026-07-14 Kaggle CPU train v1

```bash
make push-kaggle-train EXP=exp246_negative_space_gr_barrier_audit
kaggle kernels pull kentookumura/exp246-negative-space-gr-barrier-audit-train -p /tmp/kaggle-pull/exp246-negative-space-gr-barrier-audit-train -m
kaggle kernels status kentookumura/exp246-negative-space-gr-barrier-audit-train
```

- push時刻: 2026-07-14 12:31 UTC
- kernel: `kentookumura/exp246-negative-space-gr-barrier-audit-train`
- version: 1
- id_no: `127059485`
- pre-push cost: 1 diagnostic variant / LightGBM 0 config / fold 0 / booster 0 / parent-control再学習なし。
- metadata: CPU、internet disabled、run-on-push、exp072・exp115の2 kernel sources。
- push後のpullで同一canonical IDの存在を確認。初回statusは`RUNNING`。
- 実行中の通常logsは空。Kaggle CLIの既知挙動として、空ログだけを失敗根拠にせず同じkernel IDを監視する。
- 773-well本体処理後、row audit gzipのdecompressed SHA読取で`EOFError: Compressed file ended before the end-of-stream marker was reached`となった。
- 原因: `gzip.GzipFile(fileobj=raw)`はcaller-owned `raw`をcloseしない。`TextIOWrapper.close()`後も下位bufferが開いており、直後のSHA読取からgzip trailerが見えなかった。
- 修正: writer closeで`raw.flush()` / `raw.close()`を明示し、hash前にclosed assertionを追加。barrier、candidate、guard、thresholdは変更しない。

### 2026-07-14 Kaggle CPU train v2

- push時刻: 2026-07-14 12:41 UTC
- kernel: `kentookumura/exp246-negative-space-gr-barrier-audit-train`
- version: 2（同じcanonical ID）
- v1のKaggle最終status `ERROR`とpull成功を確認後にpush。
- 変更はgzip writerの下位buffer明示flush/closeとhash前closed assertionのみ。
- active variant / config / fold / boosterは`1 / 0 / 0 / 0`、parent/control再学習なし。初回statusは`RUNNING`。
- Kaggle最終status: `COMPLETE`。
- runtime: 733.672秒。
- processed: 773 wells / 3,783,989 rows。
- decision: `diagnostic_only_guard_failed`。

#### v2 safety guards

| Guard | value | limit | pass |
| --- | ---: | ---: | --- |
| true-path瞬時違反率 | 0.006421795 | ≤ 0.001 | false |
| true anchor-component survival | 0.991759968 | ≥ 0.995 | false |
| good-candidate false-prune率 | 0.361573980 | ≤ 0.001 | false |
| union oracle RMSE delta | +1.656898146 | ≤ 0.0 | false |
| worst-well union oracle delta | +77.747616063 | ≤ 0.25 | false |

- barrier supported率0.677376、no-survivor率0.278630。
- union oracleは7.434021からstrict 8.925525、fallback込み9.090919へ悪化。
- hidden-like spatial / typewell-purgedは+2.189781 / +2.204004悪化。
- well単位は改善0 / 同値138 / 悪化635。worstは`d07aed8f`。
- `likpf_mean` / `pf_ancc`のprune precision liftは1.394x / 1.301xだが、good false-pruneは0.312360 / 0.283587でhard exclusionには使えない。

#### 出力確認

- 採否記録に必要な`metrics.json`、summary、group/candidate/by-well/barrier-well集計だけをKaggle outputから取得した。3.78M-row auditは取得していない。
- candidate cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- raw file inventory SHA: `335b336bd47b56d1a58d42a77c7a511957b4582d5f06e2a4ad87e6229251fd9d`
- row audit raw / decompressed SHA: `5c50162c07f6b54d707ae234851930dc04c17aacc294914e2a182b1151aa488f` / `07d2c918e9315f3f934d30970fb7d60b8d3453aff1f62e99c7500f561b3f1597`
- group / candidate / by-well SHA: `23e2dd4469596a8236cb1239964dfe1a4407f24dc5634809a46a5d2bb8163089` / `eecedf398c1eac91755af4ec6e654403682829408fdd0630b9f497b9d35af42f` / `2d6329b09da7363ff34c49d5d273ac1c71fbb747a0c4d623ac15b4ceab04c9e1`
- 実行時config SHA: `4a0e7673cf6e4c523b4090eec353489352c3ecb50bcbf00384bbaaa709812724`。実行後はlocal configのstatusだけを`completed_guard_failed`へ更新。
- distance labelは未引用YAML値が数値化されたが、edgesと集計行は正しい。出力`40 / 20544 / 100250 / 250500 / 5001000 / 1000_plus`は順に`000_050 / 050_100 / 100_250 / 250_500 / 500_1000 / 1000_plus`。採否に影響しないため再実行しない。

## 次のアクション

5 guardsすべて不通過のためnegative-space hard barrierを閉じる。threshold grid、HMM/PF/Beam edge cut、raw-test inference、submissionは行わない。残す場合は累積pruneではなく瞬時endpoint/crossing/barrier fractionをadd-only confidence featureとして既存`topk_path_confidence_features`へ統合するだけに限定する。
