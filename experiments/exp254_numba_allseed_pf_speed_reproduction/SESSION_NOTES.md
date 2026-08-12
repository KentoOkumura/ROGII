# exp254_numba_allseed_pf_speed_reproduction セッションノート

## 目的

高優先度・基盤 backlog `numba_allseed_pf_speed_reproduction` を実装し、exp243 v3 exact PFの
all-seed JITとseed bank再集約の速度・parity・決定性を監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: 完了・不採用・branch closed
- inference / submission: disabled

## 固定実験契約

- 親: `exp243_pf_seed_medoids` v3 exact-parity candidate bank。
- PF: exp072 raw-GR Gaussian likelihood-PF。
- particles: 500、seeds: 最大128。
- seed: `sha256("likpf::train::<well>")[:16] % 2147483647 + 1 + seed_index`。
- dtype: trajectory / log-likelihood / aggregateまでfloat64、saved exp243 mean比較だけfloat32。
- process 1、Numba thread 1、GPU 0、internet off。
- representative well: eval rows 10/50/90%分位からtarget-freeに固定。
- seed grid: `1/4/16/32/64/128`。
- candidate spec grid: `1/10/100/300`。
- temperature: `3/5/8/12`、aggregation: mean / likelihood-weighted、deterministic seed subset。
- true TVT、error、oracle、CV/LB、selector、inference、submissionは禁止。

## 実装前コストガード

- active diagnostic variant: 1
- PF dynamics variant: 1
- representative wells: 3（probe）
- particles × max seeds: 500 × 128
- LightGBM config: 0
- fold: 0
- booster: 0
- parent/control retraining: なし
- GPU: なし
- raw-test inference / submission: なし

## 再現性

- `docs/06_reproducibility.md`を2026-07-15に確認した。
- exp243 row candidatesはdecompressed SHA、cluster summaryとraw horizontal/typewellはraw SHAを記録する。
- seed bank cacheはcontainer SHAだけに依存せず、array名・dtype・shape・C-order bytesのcontent SHAを主証拠にする。
- legacy/all-seed per-seed trajectory、log-likelihood、final meanをexact比較する。
- all-seed repeat、cache round-trip、warm candidate repeatのSHAをguardする。
- full workloadはprobe summary SHAと`probe_passed=true`がない限り停止する。

## コマンドログ

### 2026-07-15 作成

    make new-steering EXP=exp254_numba_allseed_pf_speed_reproduction
    make new-exp EXP=exp254_numba_allseed_pf_speed_reproduction

- steering: `.steering/20260715-exp254-numba-allseed-pf-speed-reproduction/`
- 親: `exp243_pf_seed_medoids`
- route: `pf_beam`

## 次のアクション

なし。all-seed高速化は再現せず、cached 300-candidate再集約にも現在の用途がないため、
full workload、後続実験、inference、submissionは行わない。

## 2026-07-15 実装

- 10章 / 2,000行超のcompact self-contained Jupytext trainを実装した。
- exp243 v3 row candidatesのdecompressed SHAとcluster summary SHAをhard guardし、K8 summaryの
  eval rows 10/50/90%分位から3 wellsをtarget-freeに選ぶ。
- horizontal inputは`MD/Z/GR/TVT_input`だけを`usecols`で読み、evaluation target列を読み込まない。
- raw horizontal/typewellからexp243と同じeval index、GR補間、sigma、prefix末尾surface/rate、
  TVT grid、stable seed baseを組み立てる。
- exp243 all-seed PF bodyと、同じbodyをsingle-seed Numba関数へ分けたPython legacy loopを独立実装した。
- shared interpolation compile、legacy/all-seed compile、warm PF、cache write/read、candidate再集約を
  `perf_counter`で分離計時し、process peak RSSを保存する。
- candidate 0は128-seed meanに固定し、temperature `3/5/8/12`、mean / likelihood-weighted、
  deterministic subset/offset/strideから一意な300 specを事前生成する。
- `.npz` cacheはfile SHAに加え、array名・dtype・shape・C-order bytesのcontent SHAを保存する。
- legacy/all-seed trajectory / log-likelihood / mean / ESS / resampling、saved exp243 float32 mean、
  all-seed repeat、cache round-trip、warm repeatをfail-closed guardにした。
- 773-well projectionは`measurement_kind=projection`として実測full runtimeと分離し、
  2–3分仮説はpinned probeを使ったfull workload実測まで判定しない。
- inference notebookはdisabled guardで必ず停止し、submission.csvを生成しない。

## 静的検証

    .venv/bin/python -m py_compile experiments/exp254_numba_allseed_pf_speed_reproduction/exp254_numba_allseed_pf_speed_reproduction_train.py experiments/exp254_numba_allseed_pf_speed_reproduction/exp254_numba_allseed_pf_speed_reproduction_inference.py
    .venv/bin/ruff check experiments/exp254_numba_allseed_pf_speed_reproduction experiments/exp254_numba_allseed_pf_speed_reproduction/tests/test_exp254_numba_allseed_pf_speed_contract.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp254_numba_allseed_pf_speed_reproduction/exp254_numba_allseed_pf_speed_reproduction_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp254_numba_allseed_pf_speed_reproduction/exp254_numba_allseed_pf_speed_reproduction_inference.py
    .venv/bin/pytest -q experiments/exp254_numba_allseed_pf_speed_reproduction/tests/test_exp254_numba_allseed_pf_speed_contract.py
    make validate-exp EXP=exp254_numba_allseed_pf_speed_reproduction
    make validate-template

- py_compile / Ruff / Jupytext train+inference round-trip / strict exp validation / template validation: PASS。
- exp254 contract test 5件: PASS。Numba未導入のlocal venvではdecoratorをidentity shimに置き換え、
  synthetic 4 seedsでtrajectory / likelihood / mean / ESS / resamplingのexact parityを確認した。
- repo全体pytestは一度27件PASSしたが、最終再実行時には作業範囲外のexp251 configが
  `feature_audit_only`から`train_after_feature_audit`へ変化しており、既存exp251 test 1件だけFAIL。
  exp251は変更せず、exp254固有5件を最終根拠とする。
- train notebook: 22 cells（10 code / 12 markdown）。inference: 8 cells（3 code / 5 markdown）。
- 親exp243正規trainは238行 / 6章で重いhelper依存。本実験は2112行 / 10章self-containedで、
  input preflight、PF input、両kernel、cache/warm generation、guard、保存をnotebook上へ展開した。
  同一exp helper importと`__file__`はない。

## Kaggle package

    make prepare-kaggle-notebooks EXP=exp254_numba_allseed_pf_speed_reproduction EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp254-numba-allseed-pf-speed-reproduction-probe --title 'exp254 numba allseed pf speed reproduction probe' --run-on-push --strict"

- canonical kernel: `kentookumura/exp254-numba-allseed-pf-speed-reproduction-probe`。
- metadata: private CPU、GPU/TPU/internet off、run-on-push、competition source 1、kernel source exp243 1本。
- source / loose package / bootstrap manifestのconfigとtrain source SHAは一致。
- config SHA: `cf5ea9ddd00dbf59de32c22d805f7dc9c02c6c2cb21d7e31c5fdc2f57b32d9dd`。
- train source SHA: `a1fa3b0ca2547f30f73459330090b71080cd5f3b4a5074d9b2932cc889b98c12`。
- canonical train notebook SHA: `1da89011ebaebcb4a9a905a8c30da2a91929a51054e3b0f839f9066fc0ed8046`。
- package prepareのみ。Kaggle push / run / output取得は行っていない。

## 2026-07-15 Kaggle CPU probe実行承認・push前ガード

- ユーザーが「実行してください」と明示したため、canonical Kaggle CPU probeのpushを承認済みと扱う。
- 実行対象: private CPU notebook 1本、target-free代表3 wells、active diagnostic variant 1、
  PF dynamics variant 1、particles 500、seed grid `1/4/16/32/64/128`。
- candidate spec grid `1/10/100/300`、temperature `3/5/8/12`、legacy/all-seed/cache warm比較。
- LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、
  raw-test inference / submission 0。
- credential checkerはAPI tokenなし、OAuth credentialsとlegacy keyは利用可能と確認した。
- strict exp validationとexp254 contract test 5件を再確認してPASS。
- package metadataはprivate CPU、GPU/TPU/internet off、run-on-push、competition source 1、
  kernel source exp243 1本。
- source/package config SHA `cf5ea9ddd00dbf59de32c22d805f7dc9c02c6c2cb21d7e31c5fdc2f57b32d9dd`、
  train source SHA `a1fa3b0ca2547f30f73459330090b71080cd5f3b4a5074d9b2932cc889b98c12` は一致。
- push予定: `kentookumura/exp254-numba-allseed-pf-speed-reproduction-probe` version 1。
- push前pullは403、自分のkernel list exact searchは`Not found`で、既存canonical kernelは確認できなかった。
- canonical kernel version 1を正常push。URL:
  https://www.kaggle.com/code/kentookumura/exp254-numba-allseed-pf-speed-reproduction-probe
- 同じcanonical IDで監視し、RUNNING中のlogs空やstatus API揺れを理由に再pushしない。
- push後pull成功、kernel id_no `127307789`、metadataはprivate CPU、GPU/TPU/internet off、
  competition source 1、exp243 kernel source 1本。初期statusは`KernelWorkerStatus.RUNNING`。

## 2026-07-15 Kaggle CPU probe v1結果

- canonical kernel v1は`KernelWorkerStatus.COMPLETE`。再pushなし。
- Kaggle notebook wall runtime 436.888720秒、peak RSS 683.09375 MiB。
- target-free固定3 wells: `12203f2a` 3,261 rows、`cdc31d65` 4,840 rows、
  `8f201368` 6,349 rows。合計14,450 eval rows。
- LightGBM config 0、fold 0、booster 0、GPU 0、parent/control再学習なし、
  inference / submissionなし。
- 128 seedsの3-well PF core合計はlegacy 80.897349秒、all-seed 81.754755秒。
  `legacy / all-seed = 0.98951x`で、well別ratioは0.98720–0.99052x。
- 300-candidate warm generationはwell別0.025540 / 0.038605 / 0.040417秒、合計0.104562秒。
  all-seed PF core / warm generation比は727.34 / 706.47 / 888.36倍。
- all-seed + warmの773-well行数比例projectionは21,436.315378秒（約5時間57分）。
  `measurement_kind=projection_from_three_fixed_length_quantile_wells`であり、full実測ではない。
- guardはtrajectory / log-likelihood / final mean exact、all-seed repeat SHA exact、
  saved exp243 float32 mean exact、cache round-trip exact、warm repeat SHA exactの全件true。
- probe summary SHA `4898d7f60e6639139981654c7fc9818c1e24dd83f677d031220d56bd52d1704d`を
  実行証拠として記録した。branch closure時にconfigのpinは空へ戻し、full workloadをfail-closedにした。
- input SHAはexp243 row candidates decompressed
  `0583836a76e1b9515f8289965a25b0f41cf661294a20195a3021efbcd43e32bd`、cluster summary raw
  `d3eea9f7c4ff777bed0a3c2ff2c60a076a1d49257c85b547a7b279df9735d333`で契約一致。
- artifact SHAはparity `3791f7cae1f3b94ed786f248f056eddf1fc06012123713dc4ec90dbf9a833cb8`、
  timings `2c50a0611ab13b88e5c9841a2cbfc1489080380a716b349883594f9b64b43d30`、
  speedups `f644badec870df6ff26ce7b069902ebc1aa63a3bed107a9d70eeae18beefb8a2`、
  projections `42eb9c8e9549d3a1143d09a80c908b8a11fc20d09345e7f2527601320d23b2ac`。
- Kaggle logのpandas downcasting FutureWarningとnbconvert SyntaxWarningは結果保存後のwarningで、
  guard/statusに影響しない。error / traceback / nonfiniteはない。
- full workloadは実行しない。2–3分end-to-end高速化はprobeで再現されず、warm candidate再集約は
  軽量でも現在の推論・実験に用途がないため採用しない。

## 2026-07-15 branch closure

- ユーザー指示「閉じてください」により、exp254を完了・不採用として閉じた。
- all-seedはlegacyより約1.06%遅く、PF高速化として不採用。
- 300-candidate warm再集約は既存seed bankがある場合だけ軽いが、固定集約1本を使う通常推論には不要。
- exp252のseed-medoid selectability gateも弱く、多数候補を生成する科学的根拠がない。
- `probe_summary_expected_sha256`を空に戻し、full workloadをfail-closedにした。
- 773-well full workload、追加最適化、candidate探索、inference、submissionへ進まない。
