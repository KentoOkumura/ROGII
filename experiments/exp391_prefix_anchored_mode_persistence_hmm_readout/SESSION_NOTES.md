# exp391_prefix_anchored_mode_persistence_hmm_readout セッションノート

## 目的

同一well上でexp209 posterior mean / marginal MAP / global Viterbi /
top-2 mode TVT・massと、exp226 K16 projection前後、exp263 fixed candidateを重ね、
ramp-to-persistent-offsetがposterior averaging、transition、K16 projection、
fixed blendのどこで生じるかを切り分ける。

HMM内原因が支持された場合だけ、prefix-anchor mode identityを前rowから追跡し、
別modeへ移ったpathを最終candidateから除外する。

## 現在の状態

- Route: pf_beam
- 状態: Kaggle private CPU Stage A1 version 3 FAIL_CLOSED・branch閉鎖
- CV: まだなし
- LB: まだなし
- 実装承認: 2026-07-25のユーザー指示であり
- Kaggle Stage A0 package / push / run: 承認済み・完了
- Kaggle Stage A1 package / push / run: 承認済み・version 3完了・FAIL_CLOSED
- inference / submission: scope外

## コマンドログ

### 2026-07-25

- `make new-steering EXP=exp391_prefix_anchored_mode_persistence_hmm_readout`
- `make new-exp EXP=exp391_prefix_anchored_mode_persistence_hmm_readout`
- AGENTS、`kaggle-review-exp`、`docs/agent-playbooks.md`、
  `docs/06_reproducibility.md`、親・参照実験を確認。
- steering、config、README、SESSION_NOTES、result、metrics、backlogを設計状態へ更新。
- notebook、helper、test、packageは変更していない。

未実行:

- notebook実装・変換・実行
- Kaggle package / push / train
- inference / submission
- artifact生成

### 2026-07-25 実装セッション

ユーザーの`exp391を実装してください`をimplementation-only承認として反映した。
A0 / A1 / BのKaggle実行、正規Notebook採用、package、push、inference、
submissionの承認には拡張していない。

実装:

- `exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.py`
  - 11章、3,407行のJupytext percent形式。
  - exp209 / exp270 / exp226 / exp263のSHA固定resolver、truth-free strict row join、
    32-row decoder separation event、5-fold quota付き16-well selectionを実装。
  - exp209 exact joint-state forward-backwardを維持しつつjoint posteriorをprocess-localに
    保持し、same-pass mean / marginal MAP / global Viterbi parityを実装。
  - top-2 peak / basin、prefix start priorからfirst-row basinへのtransport overlap、
    row間transition-transport overlap、mass rankに依存しないmode ID、
    merge / split ancestry、Viterbi `mode_switch_count`を実装。
  - anchor lineageだけを全rowで許可したexact masked forward-backwardにより、
    `prefix_anchor_no_switch_conditional_mean`を生成し、未解決時はwell全体を
    保存exp209 meanへfail closedする。
  - Stage A1 cause label / technical / mechanism / resource gate、
    Stage B truth-late fold / event / 1000+ / hidden-like / by-well /
    exp263 report-only formula gateを実装。
- `exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.py`
  - train-side readoutであることを明示し、inference / submissionをfail closedする。
- `experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/tests/test_exp391_prefix_anchored_mode_persistence_hmm_readout.py`
  - implementation-only flag、truth-read ledger、strict join、event merge、
    fold quota selection、mass-rank swap、start-prior anchor、tie-break、
    gradual cross-mode drift、merge / split、exp209 marginal parity、
    exp270 Viterbi parity、logical SHA、inference境界の14 tests。
- compact train / inferenceの`.ipynb`候補をJupytext変換で生成。

検証:

```bash
.venv/bin/pytest -q experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/tests/test_exp391_prefix_anchored_mode_persistence_hmm_readout.py
.venv/bin/ruff check experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.py experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.py experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/tests/test_exp391_prefix_anchored_mode_persistence_hmm_readout.py
.venv/bin/ruff check experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.py experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.py --select F821
.venv/bin/python -m py_compile experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.py experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.py experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/tests/test_exp391_prefix_anchored_mode_persistence_hmm_readout.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp391_prefix_anchored_mode_persistence_hmm_readout/exp391_prefix_anchored_mode_persistence_hmm_readout_compact_selfcontained_inference.py
make validate-exp EXP=exp391_prefix_anchored_mode_persistence_hmm_readout
```

- 専用pytest: `14 passed`
- Ruff / F821: PASS
- py_compile: PASS
- Jupytext train / inference round-trip: PASS
- strict experiment validation: PASS
- compact sourceの`__file__`依存 / same-exp helper import: 0
- 親exp209にcompact self-contained版はない。exp270 self-contained train
  2,680行・8章に対し、exp391はStage A0/A1/Bとmode lineageをNotebookセル内で
  追える3,407行・11章とした。
- 実装時点では正規train / inference Notebookを既存template scaffoldのまま
  上書きしなかった。後続のStage A0実行承認で正規trainだけ採用した。
- ローカルNotebook実行、inference、submissionは行っていない。
- 全repository test: `943 passed / 6 skipped / 2 failed`。exp391専用testは全PASS。
  FAILは未変更の既存状態であるexp296のstatus prefix期待と
  `execution.run_variant` / Kaggle CPU approval順序期待の2件。

### 2026-07-25 Stage A0実行承認

ユーザーの`実行してください。`を、直前に提示した正規train Notebook採用、
Kaggle private CPU package / push、Stage A0実行の承認として記録した。
A1 / B、inference、submissionの承認には拡張しない。

push前実行量:

- active scientific variant: 1（target-free saved-artifact census）
- HMM well runs: 0
- LightGBM config / trained fold / booster: 0 / 0 / 0
- PF / Beam / GPU runs: 0 / 0 / 0
- parent control retraining / replay: 0 / 0
- accelerator: private CPU、internet off

push前package監査:

- canonical kernel:
  `kentookumura/exp391-prefix-anchor-mode-persistence-hmm-readout-train`
- title: `exp391 prefix anchor mode persistence hmm readout train`
- metadata: private / CPU / internet off / run-on-push
- input kernels: exp209 / exp270 / exp226 / exp263 / exp115の固定5件
- canonical body 26 cells + bootstrap 1 cell
- bootstrap ZIP SHA:
  `7a56a669e4a499327628c0e77943f7f619a03e0dbb77c4632fd2013e94812efc`
- embedded config SHA:
  `fc103272f3453e75cb9a44045d4a0b31239383f9def819bb31ae05a4056e74c1`
- embedded compact train source SHA:
  `3149025acd597c90f380f29e5e0d8088d465455749b26aa51c0ff2d40c67f1aa`
- packaged Notebook SHA:
  `c044a738d47bbe228f6c7dc815a36bc133195fc1a8d52866ef22e4ff6d873759`

初回push:

- `kentookumura/exp391-prefix-anchor-mode-persistence-hmm-readout-train`
  （slug / title各55文字）へのpushはKaggle `SaveKernel 400`で停止した。
- Kaggle検索では未作成、同slugのmetadata pullも取得不可だった。実行は開始されていない。
- exp209 / exp270 / exp226 / exp263 / exp115の5 input kernelはすべてmetadata
  pullに成功したため、入力参照不可は除外した。
- Kaggleのkernel名長さ制約へ合わせ、仮説を表す`prefix anchored mode
  persistence hmm`を維持した49文字のcanonical slug
  `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`へ短縮して
  再packageする。旧slugとの重複は確認されていない。
- 上記のpackage SHAは初回400時点の履歴であり、再package後の正規SHAを別記する。

再package後の正規push監査:

- canonical kernel:
  `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
- title: `exp391 prefix anchored mode persistence hmm train`
- bootstrap ZIP SHA:
  `26107e449207c1ebf148b7593cf23f630e398399e99c492ea6b61b8908fb67e1`
- embedded config SHA:
  `baaf93cb4e2899b8f94da10fc2ce7c35fbaeb4640be095ed1cbd283a63ad2ac8`
- embedded compact train source SHA:
  `3149025acd597c90f380f29e5e0d8088d465455749b26aa51c0ff2d40c67f1aa`
- packaged Notebook SHA:
  `be7d86ead0e5fd6a92fa7f0c8737b2af73938c52b6b69e0a57ba86d401a9e5a3`
- metadata、27 cells、Stage A0実行量、5 input kernel、strict validationを再照合した。

#### Kaggle Stage A0 version 1

- kernel:
  `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
- version / numeric ID: `1 / 128527913`
- metadata: private / CPU / internet off、5 input kernels
- bootstrap: 24 support files
- Notebook表示実行量: active variant 1、Stage A0 HMM wells 0、
  LightGBM config / trained fold / booster / PF / Beam / GPUすべて0
- status: `ERROR`
- runtime: 約69秒でjoin時にfail closed
- error:
  `ValueError: exp226/exp263 fold mismatch rows=3074825`
- HMM実行、truth / error / hidden-like role read、候補生成はjoin前停止のため0。

原因監査:

- exp391設計上のreporting foldはexp226の保存済みgroup-safe fold。
- exp263 cacheの`outer_fold`はexp072 canonical foldによるcandidate-major storage
  partitionで、exp226のreporting foldとは異なる。sample `fold=0` partitionの
  757,738 rowsにはexp226 folds 0--4がすべて含まれ、fold label不一致は653,889 rows。
- 同sampleでexp263 `exp226_k16`とexp226 `tvt_pred`はfloat32保存差内
  （max abs `0.00048828066064743325`）、exp270との`id` mismatch 0、
  `md_since` max abs diff 0だった。
- よってcross-artifact fold label equalityは不正なidentity contractだった。
  reporting foldはexp226だけを使い、exp263は`id` / `well` / `well_row_idx` /
  `md_since`、5 storage partitions、primitive間identityをstrict検証する。
- threshold、event、well selection、candidate、weight、HMM設定は変更しない。
  identity contractだけを修正し、同じcanonical kernelのversion 2へ進む。

version 2 push前監査:

- relevant tests: `25 passed`（exp391専用14 testsを含む）
- Ruff / Jupytext round-trip / strict experiment validation: PASS
- metadata: private / CPU / internet off / run-on-push、5 input kernels
- canonical body 26 cells + bootstrap 1 cell
- bootstrap ZIP SHA:
  `23b64648dc5ba7531e8eb4cb1a983bde7095af6f6b0fc8aac82e003d7e503f05`
- embedded config SHA:
  `157a0c6f371566a33dcb0de3f0f21f66cac7d73bd90a2d9d841132029623c0a3`
- embedded compact train source SHA:
  `9737e4a2e0164505cd3a841780ac93dffda9b9c6246ea3f347aa367e72c59161`
- packaged Notebook SHA:
  `a4a4808eefdfbac1969611f022f86ec6de5ab1ca21ef6204534841ac64d4b69c`
- 実行量はversion 1と同じ1 census variant / 0 HMM / 0 booster。

#### Kaggle Stage A0 version 2

- kernel / version / numeric ID:
  `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train` /
  `2` / `128527913`
- status: `COMPLETE`、Stage A0 `pass`
- metadata: private / CPU / internet off、5 input kernels
- Notebook本体runtime / peak RSS:
  `112.82603644100004 sec / 4.0594329833984375 GB`
- 3,783,989 rows / 773 wells / reporting folds 0--4、duplicate / missing join 0。
- exp263 storage partitions 0--4、exp226 / exp263各well内fold inconsistency 0、
  exp270 / exp263 `id` / `md_since` mismatch 0。
- exp226 / exp263 fold label agreementは情報値`0.18741174987559425`。
  両者を同じfoldと解釈しない。
- finite path coverage `1.0`、exp270 / exp209 posterior mean max abs diff `0.0 ft`。
- decoder-separation events: `1,234`、event wells: `730`、
  fold別event `228 / 246 / 241 / 252 / 267`。
- preflight: 16 wells、fold別`4 / 3 / 3 / 3 / 3`。
- truth / error / hidden-like role reads before freeze: `0`。
- technical gates: 全14件PASS。
- Stage A0 HMM / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- input manifest SHA:
  `935d7fc6178279d846d83133a24c43d5abde90b0020f56b27683ca131ba29a6a`
- event manifest logical SHA:
  `30dae154edf9cb5bdc353378649c3fdd38bf3592000e0feabfbdc2083565cd09`
- preflight manifest logical SHA:
  `f02d5cc034b7d313fe9f3d33d1ef516f33e2d382a33679cfa7ad00164b5868ab`
- local取得ファイルSHA:
  metrics `81f68a31...c207`、summary `6c0a914b...39c0`、
  event CSV `5dc023db...870d`、preflight CSV `6abe2574...6252`。
- output全体は取得せず、metrics、summary、input/event/preflight manifestと
  CLI logだけを
  `/tmp/kaggle-output/exp391_prefix_anchored_mode_persistence_hmm_readout/train_v2`
  へ取得した。

## 変更点

- 親をexp209 absolute-TVT exact HMMへ固定。
- Stage A0は0-HMM保存artifact census。
- Stage A1はtarget-freeに固定した16 wellsのsame-posterior preflight。
- Stage Bはprefix-anchor no-switch conditional mean 1 variant / 773 HMM runs。
- `jump_used`だけでなく、stable `anchor_mode_id` / `current_mode_id` /
  `mode_switch_count`でgradual cross-mode driftも検出する。
- exp236はthreshold参照だけとし、exp270とのposterior混同を禁止。

## 再現性メモ

- seed policy: RNGなし。well / row / state / peak / basin / mode / eventをstable sort。
- stochastic components: なし。
- CPU/GPU runtime: CPUのみ、GPUなし、internet off。A1は16 wells、
  Bは773 wells、full上限30,600 sec / 25 GB。
- LightGBM config / trained fold / booster / PF / Beam: 0 / 0 / 0 / 0 / 0。
- parent control retraining / separate replay: 0。
- Kaggle kernel id:
  `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`、
  completed version 3 / numeric ID `128527913`。
- input SHA: exp209 control
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`、
  exp270 shard0
  `93e0aeac70b1e84d139aab05f9a1d6abd577d2a07388367cf2dac362e8f68b6d`、
  shard1
  `831cbfb5adfe09f98059f4e2a192d7913331f6c57c437fadc989f01e3c91aee5`
  を設計へpin。
- Stage A0 input / event / preflight SHAはversion 2とversion 3で一致。
  decoder manifest / mode ledger / prediction SHAはversion 3で記録済み。
- submission SHA: 対象外。
- rerun check: 初回成功run後もdeterministic anchorを主張せず、
  logical SHA一致のrerun後だけ昇格可能。

## 実行前に再確認する量

| Stage | variant | HMM wells | LGB configs | folds trained | boosters | PF | Beam |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1 | 1 diagnostic pass | 16 | 0 | 0 | 0 | 0 | 0 |
| B | 1 | 773 | 0 | 0 | 0 | 0 | 0 |

Stage A1がtechnical / mechanism gateをFAILしたためStage Bは閉鎖。
parent control variantを別に再実行しない。

## 次のアクション

1. Stage A1 FAIL_CLOSEDを横断記録へ反映する。
2. Stage B、inference、submissionを実行せずbranchを閉じる。

### 2026-07-25 Stage A1実行承認

ユーザーの`Stage A1に進んでください`を、Kaggle private CPUでの
Stage A1 package / push / run承認として記録した。実行量は
1 diagnostic pass / 16 HMM wells / LightGBM config 0 / trained fold 0 /
booster 0 / PF 0 / Beam 0 / GPU 0 / parent-control retraining 0。
Stage A0で凍結したevent条件、16-well選択、gate閾値を変更しない。
Stage B、inference、submissionの承認には拡張しない。

Stage A1 push前監査:

- kernel: `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
- metadata: private / CPU / TPU off / internet off / run-on-push
- input kernels: exp209 / exp270 / exp226 / exp263 / exp115の5件
- embedded config SHA256:
  `610ae9fa05f5a123e35e753f54b4d8e32afac216d90d4865714f2dc7494c6274`
- compact train source SHA256:
  `9737e4a2e0164505cd3a841780ac93dffda9b9c6246ea3f347aa367e72c59161`
- packaged notebook SHA256:
  `2dcaa944e07119fb00d4575ac48cefa350ee2c7b1febd6267b50e63804a1c3f9`
- kernel metadata SHA256:
  `bd0d0f1a6b29b1799720d418f9b9bd11ff6a6f3222c38cbbf35745ea51e746c0`
- package config/source SHAは埋め込みsupport manifestと一致。
- `run_stage=stage_a1`、A1承認true、Stage B承認false、
  inference/submission falseを照合した。

### 2026-07-25 Stage A1 Kaggle private CPU version 3

- kernel: `kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
- version / id_no / status: `3` / `128527913` / `COMPLETE`
- run stage: `stage_a1`
- workload: 1 diagnostic pass / 16 of 16 HMM wells /
  LightGBM config・trained fold・booster・PF・Beam・GPU・control replay各0
- kernel runtime: `18105.382207183 sec`（約5.03時間）
- HMM runtime合計: `18008.710256264003 sec`
- peak RSS: `4.132144927978516 GB`
- Stage A0再現:
  3,783,989 rows / 773 wells / 1,234 events / 730 event wells /
  固定16 wells、input/event/preflight logical SHA一致、全14 gate PASS
- Stage A1 status: `fail_closed`
- technical / mechanism / all pass: `false / false / false`
- PASS checks:
  decoder events、mode-ledger duplicate key、mode identity collision、projected RSS
- FAIL checks:
  same-pass parity、posterior normalization、projected runtime、
  HMM-supported event fraction、HMM-supported fold count
- max parity abs diff:
  `0.3500000000003638 ft`（posterior mean最大`0.26953125 ft`）
- max posterior normalization error:
  `2.4567824344456923e-05`
- projected 773-well runtime:
  `870045.8142557547 sec`（約241.68時間、10.07日）
- cause events:
  19 total / posterior averaging 1 / transition 0 / K16 projection 0 /
  fixed blend 3 / unresolved 15
- HMM-supported:
  `1 / 19 = 0.05263157894736842`、`1 / 5 folds`
- candidate:
  78,866 rowsすべて`candidate_fail_closed=True`、active 0
- truth / error / hidden-like reads before freeze: `0`

Stage A1 logical SHA:

- posterior row summary:
  `46b55f4a5a8d4fffca9f88c3a86d02b3d2d78f40700a8c7650004ef8eec7e2ca`
- mode ledger:
  `a15b1a88eab1dfce0fbb9fbd23fdaccdb443d3e7666e27b3cb960f1a0afbe334`
- Viterbi path ledger:
  `232706a2f14e2d3ef30b062a7259abe959929190e0112c519b47098358c67c16`
- cause labels:
  `b79b37af65c28f9e012fd6921a73eeaccfe65c612d9c11e46f582190cce5d3cc`
- candidate prediction:
  `111b91597c512afb318df919bf80c62d415ed42ba4f93e11c920d1c86b57916a`
- decoder contract manifest:
  `486370c0912ec4569b50aefb14e2c1cfcd9c1705a4d1189c178367426cfd1de6`

取得artifactのdecompressed CSV SHA:

- posterior row summary:
  `dd86c3c96189d772b0e84efbb59f11d78b9e77bdab39c9be997c2942ca468c8c`
- mode ledger:
  `ee4a56f32c2ae541e54a7798fa64d0b5d2faef1605c05e8bbe5563edb3b61a5d`
- Viterbi path ledger:
  `6a89626d744d562f0bc2d93672c431f381eb56206cf4359a531b06e230de3220`
- candidate:
  `bcf8b522094fd979f3e6f134309f95b6d3fa40e58d52174e3edbb647e43f807b`

必要なsummary、ledger、decoder manifest、candidate、logだけを
`/tmp/kaggle-output/exp391_prefix_anchored_mode_persistence_hmm_readout/train_v3`
へ取得した。Stage B先行条件を満たさないため、threshold / tolerance / matching /
fallback / blendを救済せず、Stage B、inference、submissionなしでbranchを閉じる。

記録更新後、ローカル`kaggle/train` packageは再実行防止のため
`run_stage=none` / `run_hmm=false` / `run_on_push=false`で再生成した。
この閉鎖packageはKaggleへpushしていない。config / notebook / metadata SHAは
`7f637c6e...8443` / `96387475...2fc` / `76e874ec...93a`。
