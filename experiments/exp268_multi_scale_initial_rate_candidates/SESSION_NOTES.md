# exp268_multi_scale_initial_rate_candidates セッションノート

## 目的

backlog `multi_scale_initial_rate_candidates`を、exp209 exact HMMの初期rate windowだけを変える
0-booster candidate-bank auditとして実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU shard 0/1 version 1完了、aggregate version 1 push直前
- CV / LB: まだなし / 対象外
- inference / submission: disabled

## 実行前コスト契約

- active HMM variants: 4 (`32/64/128/256`)
- operational well shards: 2
- target wells / HMM well-runs: 773 / 3,092
- LightGBM config / fold / booster: 0 / 0 / 0
- parent/control retraining: なし。exp209 `tail_n=30`を保存済みcontrolとして読む。
- GPU / raw-test inference / submission: なし / なし / なし
- exp209 v5 HMM実測11,285.868秒/variantから、各half-well shardは約22,572秒と見積もる。
- 2026-07-17 21:53 JSTにユーザーからKaggle実行の明示指示を受けた。

## 再現性

- `docs/06_reproducibility.md`を確認済み。
- exact HMMはno RNG。joblib threads 2 × Numba threads 2でも乱数系列は存在しない。
- well shardだけを`sha256("exp268::well_shard::<well>") % 2`で決める。
- exp072 cache decompressed SHA `99a3c70a...d320e1350`とexp209 HMM decompressed SHA
  `8e2f4236...7f7ae5`をaggregate時にhard guardする。
- shard gzipはraw SHAとdecompressed content SHA、aggregateはcandidate array content SHAを保存する。
- Numba parallel floating arithmeticの微小差を考慮しdeterministic submission anchorとは扱わない。
- model、raw-test prediction、submissionは生成しない。

## 実装

- 10章 / 1,709行 / 21 cellsのcompact self-contained Jupytext trainを実装した。
- canonical train、shard 0、shard 1は同じHMM kernelと監査関数を持ち、`RUN_KIND_OVERRIDE`だけを
  `aggregate` / `shard0` / `shard1`へ固定した。
- generatorはraw horizontalから`TVT`列をdropしたframeだけをHMMへ渡し、4 path凍結後にtrue TVTを付ける。
- HMM grid、rate grammar、transition、GR emission、sigma/calibration、start prior幅はexp209から固定した。
- shard cacheは4 path、std、initial rate、prefix rowsを保存し、well summaryへloglikとRMSEを保存する。
- aggregateはtail30 control、likpf reference、4新規pathをstrict id alignし、overall、distance、prefix、
  hidden-like、by-well、rate spread、pairwise duplicate、unique-best、oracle scopeを保存する。
- oracle predictionはmetric計算中だけ存在し、candidate cacheには書かない。
- inference notebookはdisabled guardで必ず停止し、`submission.csv`を生成しない。
- 同一exp helper importと`__file__`は使っていない。

親exp209正規trainは174行 / 6章でhelperへ委譲する構成。本実験はHMM kernel、shard orchestration、
aggregate診断までself-contained化したため1,709行 / 10章で、入力、変更変数、実行対象、保存先を
notebookだけで追える。

## コマンドログ

### 2026-07-17 作成・実装

    make new-steering EXP=exp268_multi_scale_initial_rate_candidates
    make new-exp EXP=exp268_multi_scale_initial_rate_candidates

- steering: `.steering/20260717-exp268-multi-scale-initial-rate-candidates/`
- experiment: `experiments/exp268_multi_scale_initial_rate_candidates/`

### 静的検証

    .venv/bin/python -m py_compile experiments/exp268_multi_scale_initial_rate_candidates/*.py experiments/exp268_multi_scale_initial_rate_candidates/tests/test_exp268_multi_scale_initial_rate_candidates_contract.py
    .venv/bin/ruff check experiments/exp268_multi_scale_initial_rate_candidates experiments/exp268_multi_scale_initial_rate_candidates/tests/test_exp268_multi_scale_initial_rate_candidates_contract.py --select F821
    .venv/bin/pytest -q experiments/exp268_multi_scale_initial_rate_candidates/tests/test_exp268_multi_scale_initial_rate_candidates_contract.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp268_multi_scale_initial_rate_candidates/exp268_multi_scale_initial_rate_candidates_train.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp268_multi_scale_initial_rate_candidates/exp268_multi_scale_initial_rate_candidates_train_variant0.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp268_multi_scale_initial_rate_candidates/exp268_multi_scale_initial_rate_candidates_train_variant1.py
    JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp268_multi_scale_initial_rate_candidates/exp268_multi_scale_initial_rate_candidates_inference.py
    make validate-exp EXP=exp268_multi_scale_initial_rate_candidates

- py_compile / Ruff F821 / Jupytext train 3本 + inference / strict exp validation: PASS。
- exp268 contract test 6件: PASS。
- `make validate-template`: PASS。
- `make test`: repo全体103件PASS。
- strict validation初回はREADMEの`検証方針` / `所見`節不足だけでFAILし、両節を追加した最終再実行でPASSした。
- 実データのローカルnotebook実行は行っていない。初回full実行はKaggle CPUを正とする。

## Kaggle package

- shard 0: `kentookumura/exp268-multi-scale-initial-rate-shard0`
- shard 1: `kentookumura/exp268-multi-scale-initial-rate-shard1`
- aggregate: `kentookumura/exp268-multi-scale-initial-rate-aggregate`
- disabled inference: `kentookumura/exp268-multi-scale-initial-rate-inference-disabled`
- 全packageはprivate CPU、GPU/TPU/internet off。shardはkernel source 0、aggregateだけexp072、exp209、
  exp115、shard 0/1の5 kernel sourcesを持つ。
- canonical / loose package / bootstrap ZIPのconfigとsourceはbytes一致。
- config SHA: `adef8b707c3961b7c71bb83f11fb0efa2e18dd3e830e2e4f55061dea51b5899c`
- aggregate source SHA: `8f3dcc818ccd6a2527c8e4e6aae1ce4152ee206c0fa162c68ae44d2413be47da`
- shard 0 source SHA: `6e2d3ac0e876296d3851ee48395b09bc9bf9740b2d9c8a884506ee0a40fcffb2`
- shard 1 source SHA: `fc200e61651834f62af93bb93ce3951ac6bbb3c8f721d8cc9b7367500bf2e334`
- 2026-07-17 21:53 JSTのremote existence事前確認では、shard 0/1とも`kaggle kernels pull -m`が
  `403 GetKernel`を返した。未pushというローカル記録と整合するためcanonical IDの初回pushへ進む。

### 2026-07-17 Kaggle CPU実行

- push直前に4 variants / 2 shards / 3,092 HMM well-runs / 0 configs / 0 folds / 0 boosters、
  control再生成なし、private CPU、GPU/TPU/internet offを再確認した。
- shard 0/1を同時にpushし、両方の生成物確認後だけaggregateをpushする。
- 2026-07-17 21:53 JSTにcanonical IDへversion 1をpushした。
  - shard 0: kernel id no `127592526`
  - shard 1: kernel id no `127592528`
- push後の`kaggle kernels pull -m`は両方成功し、private CPU、GPU/TPU/internet off、
  competition source設定を確認した。

## 次のアクション

aggregate version 1をcanonical IDへ1回だけpushし、coverage/diversity/SHAを確認する。

### 2026-07-19 aggregate実行再開

- exp292実行指示を受け、そのhard prerequisiteとしてexp268 aggregateを先に実行する。
- shard 0/1はKaggle `COMPLETE`。shard0は375 wells / 1,853,957 rows、shard1は398 wells /
  1,930,032 rowsで、unionは期待値773 wells / 3,783,989 rowsに一致する。
- active HMM variant 0、LightGBM config 0、trained fold 0、booster 0、HMM/PF再実行0、
  control/parent再学習なし、private CPU、GPU/TPU/internet offをpush前に再確認した。
- `kentookumura/exp268-multi-scale-initial-rate-aggregate`はremote pullが403で、未push記録と整合する。
- canonical aggregate packageをstrict再生成し、loose config/sourceのbytes一致を確認した。

### 2026-07-19 aggregate完了

- canonical aggregate version 1をpushし、kernel id `127887734`、Kaggle status `COMPLETE`を確認した。
- runtime 295.676秒、773 wells / 3,783,989 rows。shard 0/1のwell集合はdisjointでunionが期待値に一致した。
- tail30 direct RMSE 11.938287。best rate candidateはw128の11.895581で、gainは0.042706 ft。
- initial-rate-5 bankのoracle gainはrow 0.102358、H256 block 0.102151、whole-well 0.097314 ft。
- rate spread median 0、p90 0.02、zero-spread 423 wells。pairwise path duplicate率は58.99%から88.36%。
- aggregate prediction content SHA
  `fc18952f564dcefed8222ee30510828a4fb47f51c06a0eec5b1ddf37887ecdd1`、summary raw SHA
  `8bd2064892f7eb05392785d602e810b9aea8b686225994cd515247609370e0c6`、manifest raw SHA
  `427aa3f15c8577b38448836d3adea58ef69dcf43d6a79237f0c603b9bf04494b`を固定した。
- oracle prediction、candidate mean、selector、inference、submissionは生成していない。
- 子実験exp292がtarget-free識別性をFAIL-closeしたため、exp268 bankをdeployable candidateへ昇格しない。
