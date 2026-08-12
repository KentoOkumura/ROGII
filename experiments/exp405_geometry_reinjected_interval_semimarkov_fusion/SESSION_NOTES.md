# exp405_geometry_reinjected_interval_semimarkov_fusion セッションノート

## 目的

保存済みexp293 deployable12だけを区間semi-Markovで融合し、
docking-independent geometry再注入がexp399のwrong-mode lock-inを
解消できるかをtrain-side OOFで検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: scientific FAIL・exp405閉鎖・exp406 Stage 0解禁
- CV: exp405 `8.451060 ft` / exp263 anchor `8.238332 ft`
- LB: なし
- compact self-contained train候補 / dedicated test: 実装済み
- 正規Notebook: compact self-contained候補の採用承認済み
- Kaggle package: canonical kernel用に生成・整合性検証済み
- Kaggle run: version 2 COMPLETE・technical PASS / oracle PASS / scientific FAIL
- current-test / inference / submission: 無効
- steering:
  `docs/legacy/steering/20260726-exp405-geometry-reinjected-interval-semimarkov-fusion/`
- 正規train Notebook: compact self-contained版へ採用
- 正規inference Notebook / `settings.py`: template placeholder

## 2026-07-26 設計確定

### 根拠

- exp293 fixed12 H512 oracleは`3.683763 ft`、全fold`<4.12 ft`。
- exp263 fixed physical blendは`8.2383315465 ft`、Public LB `7.800`。
- exp297はH256 expected RMSE `8.620041 ft`でanchorより悪く、
  prefix-affine latent-registration evidenceを棄却済み。
- exp399は`11.395645678 ft`、geometry occupancy `0.052990`で、
  docking依存のgeometry復帰がwrong modeで弱くなる構造だった。
- exp370 trigger resetは13 triggers、AUC `0.499998`、0/5 foldsでFAILした。

### 固定契約

- H256 block / minimum duration H512
- local Type Well query `±55 ft / 5 ft`
- raw / rolling-21 / rolling-101 fixed weights
- exact posterior、hard decoderなし
- exp226 geometry segment-start floor 0.10、docking非依存
- real 1 endpoint + 2 deterministic negative controls
- saved candidate generation / model / booster / PF / HMM / Beam: すべて0

### 分岐

```text
exp405 technical + constrained-oracle + scientific ALL PASS
  -> same exp405 current-test implementation eligible

technically valid exp405 scientific FAIL
  -> exp405 close, exp406 fixed16 Stage 0 unlock

exp405 technical ERROR
  -> technical issueだけ修正し、同じcontractを再実行
```

## 実行量

- scientific endpoint: 1
- diagnostic negative controls: 2
- rows / wells / reporting folds: `3,783,989 / 773 / 5`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam / parent rerun: `0 / 0 / 0 / 0`
- fixed16 preflight後のfull runtime gate: `<=7,200 sec`
- CPU / workers / peak RSS: CPU / 2 wells / `<=25 GB`

## 2026-07-26 fixed16 preflight実行承認

- ユーザーの「実行してください」を、直前に再提示したfixed16 Kaggle CPU
  preflightだけの明示承認として記録した。
- 正規train Notebookへのcompact self-contained候補の採用、Kaggle package生成、
  canonical train kernelへのpush、完了監視を承認範囲とする。
- full saved-OOF、current-test実装、inference、submissionは承認範囲外で、
  flagを閉じたまま維持する。
- 実行されるactive scientific endpoint / negative controls:
  `1 / 2`
- fixed candidates / reporting folds:
  `12 / 5`
- fixed16 selector:
  `16 wells`、outer-fold内訳 `4 / 3 / 3 / 3 / 3`
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- PF / HMM / Beam / parent control rerun:
  `0 / 0 / 0 / 0`
- runtime:
  Kaggle CPU、GPU無効、internet無効、2-well workers

### Push前package検証

- kernel id:
  `kentookumura/exp405-geometry-reinjected-semimarkov-fusion-train`
- title:
  `exp405 geometry reinjected semimarkov fusion train`
- metadata:
  private / CPU / TPU無効 / internet無効 / run-on-push有効
- inputs:
  competition + exp293 frozen bank kernel + exp115 hidden-like kernel
- bootstrap ZIP test: PASS
- bootstrap内`config.yaml`とpackage直下`config.yaml`: byte一致
- fixed16 / Kaggle execution flag: 有効
- full OOF / current-test / inference / submission flag: 無効
- canonical train Notebook SHA256:
  `47b970e1afeac77109f5cbdfd0b2ae78e53ae12668a685ee605694165790c9df`
- packaged Notebook SHA256:
  `a56b9268ffbede1f01399da5d2d3bd11134daafb1e19e4dccf952f63231e3f6f`
- packaged config SHA256:
  `49224dcf18e4521dd26d5f8744ee70ef0358b450c4b5db37dab93c180b75ab95`
- bootstrap ZIP SHA256:
  `c224599983aa37c9f8bc2968e03b86d8e7988aaa91c72c65160fd450ab377c3a`

### 初回pushのmetadata 400と復旧

- 初回canonical候補
  `exp405-geometry-reinjected-interval-semimarkov-fusion-train`
  （slug / titleとも59文字）は、Kaggle `SaveKernel` の詳細なし400で未作成だった。
- input source 2件はKaggle CLIから参照可能であることを確認した。
- scientific contract、Notebook、config、runtimeは変更していない。
- Kaggleのkernel名制約に収めつつ意味のあるsuffixを維持するため、
  `interval`だけを省略した50文字の
  `exp405-geometry-reinjected-semimarkov-fusion-train`
  へid / titleを同時にそろえて再packageする。

### Kaggle fixed16 preflight version 1

- push:
  `make push-kaggle-train EXP=exp405_geometry_reinjected_interval_semimarkov_fusion`
- result:
  `Kernel version 1 successfully pushed`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp405-geometry-reinjected-semimarkov-fusion-train`
- Kaggle kernel id_no:
  `128631270`
- pull後metadata:
  private / CPU / TPU無効 / internet無効 / `machine_shape: None`
- pull後source:
  competition、exp293 frozen bank、exp115 hidden-likeが設定済み
- 状態:
  `KernelWorkerStatus.COMPLETE`
- fixed16:
  `81,485 rows / 16 wells / 12 candidates / 3 controls`
- elapsed / measured peak RSS:
  `24.578047 sec / 1.317642 GB`
- projected full runtime / peak RSS:
  `1,187.426900 sec / 1.822834 GB`
- runtime gate:
  `<=7,200 sec / <=25 GB`を両方PASS
- technical gates:
  candidate SHA、fixed16 well count、5 folds、truth/hidden pre-freeze read 0、
  posterior/row-weight normalization、finite prediction、convex hull、
  physical continuity、block-center interpolation、runtime/RSSの13項目を全PASS
- normalization max error:
  posterior `1.2212453270876722e-14`、
  row weight `4.440892098500626e-16`
- summary実ファイルを必要最小限のKaggle outputとして取得し、
  `artifacts/exp405_geometry_reinjected_interval_semimarkov_fusion_preflight_summary.json`
  へ保存した。
- summary file SHA256:
  `78774852751fcb534f528938f03006c97aecfe0c516359144f8f11cd2826a9c6`
- Kaggle metrics file SHA256:
  `88a4e518e3c93f028a6ccda53a7dc9998df0ce4c60c4eb2dde9e9c192a20200d`
- logical SHA256:
  input manifest
  `276003369f573d5ce042a8b3adf01968cf51883a2470c263877aad5c534e03d8`、
  score
  `3db157c9ba0bb5731b3d62b18e1aaf8e61b9b36187d45a0cd99475b5af4cd459`、
  posterior
  `80174b9c1950ac21f8cf5024acb26bea08ae25b23121ece4701b84d2321c9510`、
  prediction
  `773659a402f87ae8da7254d8e38790752a42da55bf564fc6a39fe835627722cd`
- fixed16はresource / leakage / numerical integrityのtechnical preflightであり、
  truth join、RMSE、constrained oracle、negative-control分離、scientific gateは
  full saved-OOFまで未評価。
- fixed16 / Kaggle execution flagを再度閉じた。full saved-OOFは
  ユーザーの別承認なしに有効化・pushしない。

### 実行後local package fail-closed

- Kaggle側version 1は変更せず、ローカルpackageだけを再生成した。
- `run_on_push: false`
- `run_stage: implementation_only`
- fixed16 / full / Kaggle execution / current-test: すべて無効
- local package Notebook SHA256:
  `f8d412c430e7eb4764be4c723185f362ba812f734f1e69a52c25cade1a5d3ce8`
- local package config SHA256:
  `805127cde1d58d96e91db294900917433a0045ba100daf2866c61d77b7e759c4`
- local metadata SHA256:
  `09f8b98451f4db051d0bf2c75e7d877c06127713de5e2bcd1fd3d4e8b640bd61`
- local bootstrap ZIP SHA256:
  `a50db8e7f38243babcc4355c194e6bd49e336165a9080f22d8a1e2e20fb93d0c`

## 2026-07-26 full saved-OOF実行承認

- ユーザーの「full oofを実行してください」をfull saved-OOFの明示承認として
  記録した。
- fixed16 summary SHA
  `78774852751fcb534f528938f03006c97aecfe0c516359144f8f11cd2826a9c6`
  とtechnical gate 13/13 PASSを前提証拠とする。
- 実行量:
  `3,783,989 rows / 773 wells / 12 fixed candidates / 3 controls /
  1 scientific endpoint / 5 reporting folds`
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- PF / HMM / Beam / parent control rerun:
  `0 / 0 / 0 / 0`
- fixed16投影:
  `1,187.426900 sec / 1.822834 GB`
- runtime:
  Kaggle CPU、GPU無効、internet無効、2-well workers、
  hard gate `<=7,200 sec / <=25 GB`
- 同じcanonical kernel
  `kentookumura/exp405-geometry-reinjected-semimarkov-fusion-train`
  のversion追加として実行する。
- current-test実装、inference、submissionは承認範囲外で無効のまま維持する。

### full OOF push前package検証

- existing kernel pull:
  id_no `128631270`、version 1のsame canonical kernelを確認
- metadata:
  private / CPU / TPU無効 / internet無効 / run-on-push有効
- bootstrap ZIP test / embedded config byte一致: PASS
- full stage / approval / fixed16 evidence / Kaggle execution flag: 有効
- current-test / inference / submission: 無効
- execution counts:
  endpoint `1`、controls `2`、folds `5`、candidates `12`、
  model/config/trained-fold/booster/PF/HMM/Beam/parent rerun `0`
- packaged Notebook SHA256:
  `7aa72c748ca16d53e5c118230850a1c6a6c2d89aa9e62ef9e8ea4aa0ee266a40`
- packaged config SHA256:
  `0570313704a5b2aae64a9d689837885aaf8ab268710744630a38cfe08546fa67`
- metadata SHA256:
  `5b3bd681f11fbe7f9aad76dd98e4e70c99871e3bc8b594376341dc77675a292e`
- bootstrap ZIP SHA256:
  `8963533b0b689c10a153d9af4ca96864dd105edc7d8e735b19fa66163a62c843`

### Kaggle full saved-OOF version 2

- push:
  `make push-kaggle-train EXP=exp405_geometry_reinjected_interval_semimarkov_fusion`
- result:
  `Kernel version 2 successfully pushed`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp405-geometry-reinjected-semimarkov-fusion-train`
- Kaggle kernel id_no:
  `128631270`
- pull後metadata:
  private / CPU / TPU無効 / internet無効 / `machine_shape: None`
- pull後source:
  competition、exp293 frozen bank、exp115 hidden-likeが設定済み
- 状態:
  `KernelWorkerStatus.COMPLETE`
- elapsed / peak RSS:
  `1,434.099051 sec / 2.220737 GB`
- pooled RMSE:
  exp405 `8.451059619 ft`、exp263 anchor `8.238331745 ft`、
  delta `+0.212727874 ft`
- fold delta vs anchor:
  `+0.327486 / +0.131317 / +0.297318 / +0.051511 / +0.261915 ft`
  で0/5 folds改善
- constrained oracle:
  `3.606821673 ft`、pooledと5/5 foldsをPASS
- scope delta vs anchor:
  `1000_plus +0.230850 ft`、
  `hidden_like_spatial +0.373085 ft`、
  `hidden_like_typewell_purged +0.354028 ft`
- by-well:
  delta p95 `+1.744584 ft`でFAIL、worst regression `+3.515814 ft`はPASS
- negative-control gain:
  circular `0.000346 ft`、block permutation `0.000230 ft`。
  要求`>=0.05 ft`に届かず、foldも両controlとも3/5のみ。
- geometry:
  pooled mean `0.102489`、well median `0.102618`、
  low-mass well fraction `0.0`で3 gate PASS
- gate:
  technical 17/17 PASS、constrained-oracle 2/2 PASS、scientific FAIL、
  `all_gates_pass: false`
- decision:
  `scientific_fail_close_exp405_unlock_exp406_stage0`
- decision SHA256:
  `e159cfb712a6ed81e78f4524febbf0d995375124a473a5056aad3c1347b648f0`
- summary / gate / Kaggle metrics file SHA256:
  `9992612ba22cb615e3fd01795450c5b44bcf9cc24dcf8d6b7b331b3579d7bc77` /
  `d601df56ca58ec137a67a622c412750bf3a7a6fc455440d7823c5e14885947bf` /
  `64206258559e86efa8494819f8ceca151c4d7c47215b2a1e17eb3c025c3ee444`
- target-free logical SHA256:
  score
  `7b6f08efc27f2245b48995235a6dfca4ea06aa3d9035385251cfdef85c1920d9`、
  posterior
  `598690cb6645f692397e3dbe2ad98c469a2040a5dfb2b45152bec0c30ab908e6`、
  prediction
  `02245e4c08e7c93de82cf16051a412ecad51c2dbb0114bd237733b4d78fd41b4`
- prediction decompressed SHA256:
  `136f4e0e65c8df9e2e22bc94573948f99c1aa5cf5aa8cd00d6abc248cca2add1`
- summary、gate、fold/scope/by-well、geometry、negative-control、
  input/role ledger、SHA manifestの実ファイルを
  `experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/artifacts/`
  へ保存し、SHA manifestとraw file SHAを照合した。
- 解釈:
  candidate bankのoracle headroomは強いが、real morphology evidenceは
  negative controlsとほぼ区別できず、正しいpathを選べなかった。
- 分岐:
  same-OOF rescueなしでexp405を閉じ、exp406 fixed16 Stage 0を解禁する。

### full OOF実行後local package fail-closed

- Kaggle側version 2は変更せず、ローカルpackageだけを再生成した。
- `run_on_push: false`
- `run_stage: implementation_only`
- fixed16 / full / Kaggle package / push / execution / current-test /
  inference / submission: すべて無効
- full OOFの完了証跡:
  kernel version `2`、summary / gate / decision SHAをembedded configへ保持
- local package Notebook SHA256:
  `92518cdf9b9c96abd9fcc4ca6fe7e747b32a2d7ab4c13fecc6722af85afbcba3`
- local package config SHA256:
  `73a8c3d8d85d76ff15f8993e5816d1b0e4984b9505906fbb80313ae0f747e1a9`
- local metadata SHA256:
  `09f8b98451f4db051d0bf2c75e7d877c06127713de5e2bcd1fd3d4e8b640bd61`
- local bootstrap ZIP SHA256:
  `06963f9e879e47aa686cff97761dfcf0224ac81ef281f24bdfad90e4ef8f85f7`
- bootstrap manifest 23 files / embedded config byte一致: PASS
- このfail-closed packageはKaggleへpushしていない。

## 2026-07-26 implementation-only

### 実装

- ユーザーの「exp405を実装してください」をimplementation-onlyの明示承認として、
  正規Notebookを上書きせず、別名Jupytext percent形式のcompact
  self-contained train候補を実装した。
- exp293 candidate matrix / manifest / block assignmentをraw / decompressed /
  logical SHA、row / well / fold、candidate順で検証する。
- horizontalはpre-truthに`MD / GR / TVT_input`だけを読み、truth-bearing raw file
  SHAもprediction freeze後まで遅延する。hidden-like roleもfreeze後だけ読む。
- H256 block内centered full-window raw / rolling-21 / rolling-101を
  `0.50 / 0.25 / 0.25`で固定合成し、23 shiftsをLaplace priorで周辺化する。
  reliable candidate likelihoodへ20%のcandidate-common uniform mixtureを加える。
- circular controlはfinite GRだけをSHA256固定rotationしrow nan maskを維持する。
  block-order controlはfull H256 blockだけをSHA256順で並べ替え、最終short
  blockを固定する。
- exact semi-Markov forward-backwardはminimum duration 2 blocks、
  final short right-censor、uniform duration、log9 switch penalty、
  current state非依存のgeometry floorを実装した。hard decoderは作らない。
- block posteriorをblock center間で線形補間し、row weight正規化、candidate
  convex hull、導出したstep boundによるphysical continuityを検証する。
- score / posterior / 3 prediction / pretruth input / role ledgerをfreezeし、
  その後だけsuffix truthとhidden-like roleを読み、duration-constrained oracle、
  fold / scope / by-well / negative-control / geometry-mass gateを評価する。
- fixed16はouter fold別SHA256 rankのround-robinで`4 / 3 / 3 / 3 / 3` wellsを
  固定する。preflight / fullは別承認flagがない限り実行不能にした。

### 実装時の実行量確認

- active scientific endpoint: 1
- negative controls: 2
- reporting folds: 5
- fixed candidate paths: 12
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam / parent rerun: `0 / 0 / 0 / 0`
- current-test / inference / submission: 0

### Notebook比較

- 親exp293 compact train: 8章 / 1,963行
- exp405 compact train候補: 10章 / 約2,750行
- exp405は親のpath/SHA、input、truth-late、readout、generated-outputの役割を保持し、
  morphology、negative controls、exact semi-Markov、fixed16 resourceを追加した。
- 同一exp helper import、`__file__`、薄い`main()`だけの構成は使用していない。

### 検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/*compact_selfcontained*_train.py \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/tests/test_exp405_geometry_reinjected_interval_semimarkov_fusion.py
.venv/bin/ruff check \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/*compact_selfcontained*_train.py \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/tests/test_exp405_geometry_reinjected_interval_semimarkov_fusion.py \
  --select F821,F811
.venv/bin/pytest -q \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/tests/test_exp405_geometry_reinjected_interval_semimarkov_fusion.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/exp405_geometry_reinjected_interval_semimarkov_fusion_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp405_geometry_reinjected_interval_semimarkov_fusion/exp405_geometry_reinjected_interval_semimarkov_fusion_compact_selfcontained_train.py
make validate-exp EXP=exp405_geometry_reinjected_interval_semimarkov_fusion
make validate-template
make test
```

- py_compile: PASS
- Ruff F821 / F811: PASS
- dedicated pytest: `11 passed`
- Jupytext変換 / round-trip: PASS
- strict `make validate-exp`: PASS
- `make validate-template`: PASS
- 正規Notebook: 上書きなし
- Kaggle package / execution: なし

全体回帰`make test`は`1,149 passed / 7 skipped / 6 failed`だった。
exp405専用11件はこの全体実行内でもPASSした。失敗はexp405外の既存状態で、
exp293 downstream contract本文の現SHAとhistorical hard-coded SHAの不一致2件、
完了後configへ対して実行前status/flagを期待するexp296のstale test 2件、
全体実行中に更新されたexp403 config/testの一時不整合2件だった。
直後の再実行ではexp403は全件PASSし、exp293 / exp296の4件だけが再現した。
exp405実装のためにこれら他実験のhistorical contractや完了状態を巻き戻していない。

## コマンドログ

design-only時点で実行したのはscaffold作成とdesign文書編集だけ。

```bash
make new-steering EXP=exp405_geometry_reinjected_interval_semimarkov_fusion
make new-exp EXP=exp405_geometry_reinjected_interval_semimarkov_fusion \
  SOURCE=templates/experiment
```

implementation-onlyでは静的検証とsynthetic testだけを追加した。
学習、推論、Kaggle、local Notebook実行は行っていない。

## 再現性メモ

- real endpoint: RNGなし
- negative control:
  `SHA256("exp405::<control>::<well_id>")`でrotation/permutationを固定
- parallel: 2-well threads、global RNGなし、immutable keyで再sort
- runtime: Kaggle CPU / GPUなし / internet offを将来の正とする
- input SHA: exp293 candidate content `294771...b474`、
  block decompressed `b0755c...32d7`
- fixed16とfull OOFのscore / posterior / prediction logical SHAを
  各summaryに記録した。
- model / submission SHA: fitted modelとsubmissionを作らないため対象外
- deterministic anchor: false

## 禁止事項

- candidate、block、shift grid、weight、duration、prior、gateの変更
- exp297 evidence、exp399 docking transition、exp370 trigger resetの再利用
- hard top1 / Viterbi / row-wise switch / ML selector
- same-OOF rescue
- exp405 full saved-OOF再実行、current-test、inference、submission

## 次のアクション

exp405はclosed。current-test、inference、submissionへ進まない。
独立familyの`exp406_loop_closed_multiwell_rgt_fixed16_stage0`は
実装可能状態へ解禁したが、実装・Kaggle実行は別のユーザー承認後だけ行う。
