# exp435_tvt_memoryless_u_rate_dzonly_hmm セッションノート

## 目的

TVT確率分布だけを持続状態とする41-rate memoryless HMMと、
同一kernelのdz-only `r_U=0`を同時比較し、rate履歴と非ゼロrate supportの効果を
分離する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage0_fail_closed_all_variants`
- CV / LB: なし
- implementation: ユーザー指示により完了
- Kaggle package / push / run: Stage 0 version 1 COMPLETE
- inference / submission: 未承認、無効

## 2026-07-29 設計記録

- 作業開始時は実験ディレクトリがexp433までだったためexp434 scaffoldを作成したが、
  並行更新で`exp434_physics_candidate_public_lb_audit`が登録・作成されたことを検出した。
- 番号重複を避け、今回の実験を未使用のexp435へ繰り下げた。
- 科学的親をexp209、原因証拠をexp408に固定した。
- exp424のmomentum単独FAILとexp355のnonzero-rate平均signal / tail FAILを参照した。
- scientific treatmentを`memoryless_41rate`と`dz_only_r0`の2本に固定した。
- 両treatmentはTVT posterior mean 1点ではなくTVT確率分布を次行へ伝える。
- memorylessの41 rate重みはzero-centered parent-AR stationary分布へ固定した。
- dz-onlyは同一position-only kernelのdelta-at-zero特殊ケースへ固定した。
- Stage 0は2 treatment × fixed32 = 64 HMM well-runs、parent rerun 0。
- Stage 1はeligible treatmentごとに773 wells、最大1,546 HMM well-runs。
- model / LightGBM config / trained fold / booster / PF / Beam / GPUはすべて0。
- Stage 0、Stage 1、inference、submissionは各段階で別承認を必要とする。
- `docs/06_reproducibility.md`を確認し、RNGなし、truth-late、logical SHAを固定した。

## コマンドログ

```bash
task new-steering EXP=exp434_tvt_memoryless_u_rate_dzonly_hmm
task new-exp EXP=exp434_tvt_memoryless_u_rate_dzonly_hmm
```

- 環境に`task`がなく、上記は`command not found`でファイル作成前に停止した。

```bash
make new-steering EXP=exp434_tvt_memoryless_u_rate_dzonly_hmm
make new-exp EXP=exp434_tvt_memoryless_u_rate_dzonly_hmm
```

- Makefileの同等手順でscaffoldを作成後、番号競合を検出してexp435へ移した。

## 2026-07-29 実装記録

- ユーザーの`exp435を実装してください`をimplementationと正規Notebook採用の
  明示指示として記録した。Stage 0 package / push / runの承認には拡張しない。
- `exp435_tvt_memoryless_u_rate_dzonly_hmm_compact_selfcontained_train.py`
  をJupytext percent形式で実装した。
- persistent stateは`(time, TVT grid)`だけとし、rate軸をalpha / betaへ持たない。
- `memoryless_41rate`は親互換per-well symmetric supportと
  `sig_r / sqrt(1 - mom^2)`のzero-centered stationary重みを固定した。
- 各rateの5-cell position kernelを固定重みで集約し、TVT forward-backwardへ渡す。
- `dz_only_r0`は同じ`run_tvt_only_hmm`へ`rates=[0.0]`,
  `weights=[1.0]`を渡す特殊ケースとして実装した。
- edge-rate readoutは各行で使用した固定priorのmean / std / edge massであり、
  filtered / smoothed rate posteriorではない。次行stateには保持しない。
- 保存exp209 predictionは比較用load-onlyとし、parent HMM rerunは0に固定した。
- fixed32 manifestはprediction前に`well / prefix_rows / suffix_rows`だけ読み、
  role / foldは全32 wells × 2 variantsのprediction / diagnostic freeze後に再読する。
- truth、persistent episode、exp408 exclusive causeも全freeze後にだけjoinする。
- exp408 episode cause artifactを
  `b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`
  へSHA固定した。
- variant別mechanism gateは独立判定し、一方のFAILで他方の評価を省略しない。
- fail-closed inference guardを実装し、正規train / inference Notebookへ採用した。

### 親compact比較

- 科学的親exp209にはcompact self-contained版がないため、同じexp209 exact-HMM
  fixed32系のexp424 compactを構成参照にした。
- exp424 / exp435 train sourceは`2,239 / 2,452 lines`。
- 両方とも9章構成で、exp435はImports、notebook-safe path / SHA、
  fixed32 / saved parent input、HMM input、route-specific kernel、
  target-free freeze、truth-late readout、gate / 生成物、guarded orchestrationを持つ。
- 同一exp内helper import、`__file__`、薄い`main()` entrypointはない。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp435_tvt_memoryless_u_rate_dzonly_hmm/*compact_selfcontained*.py \
  tests/test_exp435_tvt_memoryless_u_rate_dzonly_hmm.py
.venv/bin/ruff check \
  experiments/exp435_tvt_memoryless_u_rate_dzonly_hmm/*compact_selfcontained*.py \
  tests/test_exp435_tvt_memoryless_u_rate_dzonly_hmm.py --select F821
.venv/bin/pytest -q tests/test_exp435_tvt_memoryless_u_rate_dzonly_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp435_tvt_memoryless_u_rate_dzonly_hmm/\
exp435_tvt_memoryless_u_rate_dzonly_hmm_compact_selfcontained_train.py
make validate-exp EXP=exp435_tvt_memoryless_u_rate_dzonly_hmm
```

- py_compile: PASS
- Ruff F821: PASS
- 専用test: `11 passed`
- Jupytext train / inference round-trip: PASS
- strict experiment validation: PASS
- ローカルNotebook実行、Kaggle package / push / run: 未実施

## 実行量契約

- Stage 0 treatment variants / wells per variant / HMM well-runs:
  `2 / 32 / 64`
- Stage 0 parent HMM reruns: `0`
- Stage 1 maximum variants / wells per variant / HMM well-runs:
  `2 / 773 / 1,546`
- Stage 1 parent HMM reruns: `0`
- LightGBM configs / trained folds / boosters / model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 2026-07-29 Stage 0実行承認

- ユーザーの`実行してください。`を、直前に実装したexp435のKaggle private CPU
  Stage 0 package / push / run承認として記録した。
- 承認対象はfixed32 mechanism preflightのみ。Stage 1、inference、
  submissionは引き続き未承認で無効。
- push前の実行量を次のとおり再確認した。
  - active scientific variants: `2`
  - treatment variants / wells per variant / HMM well-runs: `2 / 32 / 64`
  - saved exp209 parent HMM reruns: `0`
  - LightGBM configs / trained folds / boosters / models: `0 / 0 / 0 / 0`
  - PF / Beam / GPU runs: `0 / 0 / 0`
- runtimeはKaggle private CPU、internet無効、submission生成なし。
- canonical kernel id / title:
  `kentookumura/exp435-tvt-memoryless-u-rate-dzonly-hmm-train` /
  `exp435 tvt memoryless u rate dzonly hmm train`

### Package / push記録

```bash
make prepare-kaggle-notebooks \
  EXP=exp435_tvt_memoryless_u_rate_dzonly_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp435-tvt-memoryless-u-rate-dzonly-hmm-train \
  --title 'exp435 tvt memoryless u rate dzonly hmm train' \
  --run-on-push --strict"
make push-kaggle-train EXP=exp435_tvt_memoryless_u_rate_dzonly_hmm
```

- 初回packageはcanonical id/title、CPU / internet無効、exp209 kernel source、
  3 bootstrap asset SHAをすべて満たしたが、Kaggle `SaveKernel`が詳細なしの
  HTTP 400を返した。
- canonical slugは45文字でtitle由来slugと完全一致し、同IDのpullは500、
  mine listにもnotebookは現れなかったため、実行version作成は確認できない。
- self-contained notebookはrepo `src/`をimportしない。一方、初回packageは
  未使用のrepo `src/`全体をbootstrapへ含めて1,310,622 bytesになっていた。
  科学contract、入力、実行量、id/titleを変えず、`--no-src`で未使用payloadだけを
  除外して同じcanonical idへ再package / pushする。
- `--no-src` packageは907,343 bytesとなり、同じcanonical id/titleへのpushに成功。
  Kaggle private CPU version 1を2026-07-29 10:58:22 UTCに開始した。
- kernel id / version / id_no:
  `kentookumura/exp435-tvt-memoryless-u-rate-dzonly-hmm-train` / `1` /
  `129049294`
- package notebook / metadata SHA256:
  `bc8b973d16a68f693ae4fbc956b223a85cec1f84b7b098f3f9bf3b1507d324af` /
  `d8feeba82093f547981abcdff24ec1c87814e300e2999efbe41fc222a50111b5`
- push後metadata read-back:
  private、CPU、internet無効、competition sourceとexp209 kernel sourceが一致。
- 初回status確認:
  `KernelWorkerStatus.RUNNING`

## 2026-07-29 Kaggle Stage 0 version 1結果

- kernel:
  `kentookumura/exp435-tvt-memoryless-u-rate-dzonly-hmm-train`
- version / id_no / status:
  `1 / 129049294 / COMPLETE`
- Stage 0 notebook elapsed / peak RSS:
  `46.077013096 sec / 0.455474854 GB`
- full Stage 1 maximum runtime projection:
  `379.764049737 sec <= 30,600 sec`
- execution:
  `2 variants × 32 wells = 64 HMM well-runs`、parent rerun 0、
  LightGBM / model / booster / PF / Beam / GPU各0
- fixed32:
  32 wells、156,088 suffix rows。mechanism-onlyでありCV / promotion evidenceではない。
- logsを根拠に評価し、Kaggle output archiveは取得していない。

### Technical gate

- 結果: 全項目PASS
- finite coverage: `1.0`
- transition row-sum max error: `4.440892099e-16`
- posterior normalization max error: `3.330669074e-15`
- dz delta-at-zero parity max abs: `0.0 ft`
- truth / role-fold / episode reads before all-variant freeze:
  `0 / 0 / 0`
- TVT-only persistent state、persistent rate state cells `0`、
  rate responsibility non-persistence: PASS
- runtime projection / peak RSS / prediction and diagnostic readback SHA: PASS

### Variant別mechanism gate

| variant | forward-cause SSE reduction | persistent SSE reduction | improved wells | improving folds | matched-control pooled delta | control p95 delta | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `memoryless_41rate` | `27.205050%` PASS | `11.244859%` PASS | `4/16` FAIL | `1/5` FAIL | `+16.151527 ft` FAIL | `+29.129905 ft` FAIL | FAIL |
| `dz_only_r0` | `43.429062%` PASS | `21.835503%` PASS | `5/16` FAIL | `1/5` FAIL | `+13.705216 ft` FAIL | `+24.955652 ft` FAIL | FAIL |

- matched-control parent RMSEは`3.428436286 ft`。
- candidate matched-control RMSEはmemoryless / dz-onlyで
  `19.579963225 / 17.133652291 ft`。
- 両variantともforward-cause / pooled persistent episode SSEは改善したが、
  改善は1 foldへ集中し、persistent well多数とmatched controlを大幅に壊した。
- `stage0_all_variants_pass=false`、
  Stage 1 eligible variantは空。
- decision:
  `stage0_fail_closed_all_variants`

### 生成物SHA（Kaggle logs）

- scientific contract:
  `92f3e307007fa1dc94bd4921f519aa01267f044c0874b31d6581a61a7a356a63`
- input manifest:
  `53a918ba6b7b7fb535cc9358a6402b4e9347bee12d0329d81fe2ed70b05e7950`
- predictions logical / readback:
  `aa79810f6c189dd7fbb9d53b8c172a4a051d29ac1780ee4696237e8c24e214c3`
- rate readouts logical / readback:
  `1a554d22071e4d9210808ccbbd6f326257fa5e6265b4e117d4116c4c394f0495`
- episode truth-late readout:
  `a4e50c688dd64eb6a40e2575fff7e94f622ce4f84df16847837861056f22b0ae`
- well metrics:
  `33abe461c48170a21d44084c539e5dc1b1d9a639dab75efe18a4515a6e98e302`
- summary file:
  `469e96de487672396575a3969fad71ab1aaf9c7b16c6d1cf714d2a5d9f62e534`
- memoryless prediction / diagnostic manifests:
  `27b323e787a8236a132efa9294aba763d273703e656d34fb597db4c82e26ec66` /
  `c081122f49543e897a5fdfeb0d29a38c76c70fa57c12b5fd6a06baaad6879ccd`
- dz-only prediction / diagnostic manifests:
  `7b14d500d0f593fc82d27b3911e52ee2362c66aa9d9193123e8932ee64afec7b` /
  `19a2a2e0743ec3b7b4036c347581df3534600c38f866b6887b9d618d9958e721`

### 解釈と判断

- rate履歴除去は、既知のforward-hysteresis episodeのSSEを軽減するsignal自体は
  示した。
- ただしTVT-only transitionはmatched controlで桁違いのnegative transferを起こし、
  well / fold再現性も不足した。非ゼロstationary rate supportもdz-onlyより安全ではない。
- 数値異常、runtime、leakage、SHAの問題ではなく、固定科学介入そのもののFAIL。
- rate重み、support、noise、emission、grid、gate、blend、selectorの救済は行わない。
  Stage 1、inference、submissionへ進めずbranchを閉じる。

### 結果反映後の検証

- `metrics.json`、`result.md`、`README.md`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`、steering tasklistを最終判定へ更新した。
- 再実行防止のためlocal configの`execution.run_hmm` /
  `execution.create_prediction`をfalseへ戻した。実行済みKaggle packageは
  version 1のread-back証拠として保持した。
- inference guardをStage 0完了・両variant FAIL状態へ同期し、正規inference
  Notebookを再生成した。inference / submissionは引き続きfail-closed。
- dedicated tests `11 passed`、strict experiment validation PASS、
  template validation PASS、inference Jupytext round-trip PASS。

## 再現性メモ

- seed policy:
  RNGなし、well / row / TVT grid / rate candidate / variant順を固定
- stochastic components:
  なし
- CPU/GPU runtime:
  実行する場合はKaggle private CPU、GPU / internet無効
- truth-late:
  全candidate prediction / diagnostic SHA freeze後にtruth、fold、role、
  episode、errorをjoin
- input SHA:
  exp209 saved prediction、fixed32 manifest、exp408 ledger、exp226 / exp263、
  hidden-like assignmentを記録予定
- output SHA:
  transition contract、prediction、posterior std、report-only edge-rate readout、
  metricsのlogical SHAを記録予定
- deterministic anchor:
  submissionを生成しないためfalse

## 禁止事項

- uniform / prefix-centered / row-adaptive rate weightの追加
- rate count / span、noise、GR sigma、position grid、gateのsame-OOF変更
- TVT posterior mean 1点だけを次行へ渡す簡略化
- suffix truth TVTによる再anchor
- blend weight探索、well / row selector、inference、submission

## 次のアクション

両variantをStage 0 FAILとして閉じる。Stage 1、inference、submission、
同一OOFでのparameter / gate / blend / selector救済は行わない。今回の失敗から
独立した新しい根拠がない限り、同familyの後続backlogは追加しない。
