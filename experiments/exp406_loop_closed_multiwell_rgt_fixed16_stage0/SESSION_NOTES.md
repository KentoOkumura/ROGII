# exp406_loop_closed_multiwell_rgt_fixed16_stage0 セッションノート

## 目的

exp405がscientific FAILとなった場合の次候補として、
GR-first loop-closed multi-well RGTを固定16 wellsのStage 0で
反証可能に検証できる設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU fixed16 Stage 0 version 1 technical FAIL・branch閉鎖
- blocked by: terminal decision `close_exp406_without_parameter_rescue`
- CV / LB: なし
- compact self-contained train / fail-closed inference候補: 実装済み
- 正規train Notebook: compact self-contained候補を採用
- fixed16 Stage 0 run: 完了後に無効化
- full OOF / current-test / inference / submission: 無効

## 2026-07-26 Kaggle version 1

- kernel:
  `kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train`
- version / id_no: `1 / 128637170`
- pushed: `2026-07-26 02:29:14 UTC`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train`
- Kaggle pullでprivate / CPU / internet off、exp065/226/405 sourceを確認。
- push後にユーザー指示でpollingを停止し、完了連絡後に同じkernelを確認。
  final statusは`KernelWorkerStatus.COMPLETE`。再pushなし。

### Version 1結果

- elapsed: target-free `1,356.649085 sec` /
  total diagnostic `1,356.666020 sec`
- technical: `12/15` PASS
- FAIL:
  - graph query coverage `0.451157 < 0.90`
  - finite loop-closed row coverage `0.755026 < 0.95`
  - projected runtime `65,543.108929 sec > 30,600 sec`
- PASS:
  - connected target coverage `1.0`
  - fundamental cycles `9,272`
  - raw / solved cycle residual p95 `70.0 / 7.1e-15 ft`
  - cycle p95 reduction `1.0`
  - peak RSS `0.544994 GB`
- negative control: real-circular NCC `+0.874148`、5/5 foldsでreal優位
- fixed16 well-level:
  - graph query gate達成 `0/16`
  - finite row coverage gate達成 `5/16`
  - query coverage min / median / max:
    `0.127660 / 0.458698 / 0.739130`
- target rejection countの主因:
  - nonpositive local TVT progress `43.97%`
  - NCC threshold未満 `19.79%`
  - finite pair不足 `16.24%`
  - retained `0.63%`
- target-free gateで停止したためprefix truth joinなし、
  exp226 fixed16 K16 replay `0`、prefix科学性能は未評価。
- source-target overlap / suffix truth / target Formation / hidden roleの
  freeze前readは全0。unknown suffix predictionなし。
- decision:
  `close_exp406_without_parameter_rescue`

### 生成物監査

- Kaggle outputから小規模Stage 0生成物を`artifacts/`へ保存した。
- pairwise edges `14,066` rows、cycle basis `9,272` rows、
  loop gauge `4,811` rows、role-read ledger `191` rows。
- manifest記載file SHAを取得ファイルで`8/8`照合した。
- gauge columnsに`solved_tvt`、truth、Formation列なし。
- summary SHA256:
  `e9332c3166c875ee663c41d3f7bec0d17c68f3902aa9eb372996d77920575413`
- gate SHA256:
  `677613e1d8d856a93e2a5d1fd65840cfd8dc80064facd503713454a042f0afb8`
- SHA manifest SHA256:
  `92f53159a550dfba0a37c4a657e7cb98660357d9279e56783062201ae2315321`

## 2026-07-26 exp405分岐確定

- exp405 canonical Kaggle CPU kernel version 2（id_no `128631270`）は
  technical 17/17 PASS、constrained oracle 2/2 PASS、scientific FAIL。
- decision:
  `scientific_fail_close_exp405_unlock_exp406_stage0`
- decision SHA256:
  `e159cfb712a6ed81e78f4524febbf0d995375124a473a5056aad3c1347b648f0`
- exp406 Stage 0の開始条件は成立した。
- これは実装・Kaggle実行の承認ではないため、すべての実行flagは無効のまま。

## 2026-07-26 設計確定

### 開始条件

`exp405_geometry_reinjected_interval_semimarkov_fusion`がtechnical gateを通過し、
scientific gateの1つ以上でFAILしたdecisionとSHAが存在するときだけ開始する。
technical ERRORや未実行はexp406を解禁しない。

### 固定Stage 0

- exp386と同じfixed16 selector、5 reporting folds
- target observed inputs: `MD/X/Y/Z/GR/TVT_input`
- target Formation / suffix TVT / hidden role: freeze前read 0
- H256 / H128、±55 ft / 5 ft、12 donors、top4 edges
- GR morphology raw / roll21 / roll101
- TVT-ft loop closure、Huber delta 5 ft、10 IRLS
- circular controlとprefix512 rolling-origin
- unknown suffix prediction / scenario / model / PF / HMM / Beam: 0

### 合否

- graph query `>=0.90`
- connected target coverage 1.0
- finite loop-closed RGT coverage `>=0.95`
- cycles `>=30`、cycle p95 `<=5 ft`、raw比50%以上減
- real NCCのcircular差`>=0.10`、4/5 folds
- prefix RMSEのexp226比改善`>=0.25 ft`、4/5 folds
- full投影`<=30,600 sec / <=25 GB`

全AND PASS時だけfull-OOF Stage 1の設計資格を得る。FAIL時は同じStage 0で
donor/window/shift/edge/Huber/gateを調整しない。

## 実行量

- Stage 0 diagnostic: 1
- target wells / graph contexts: `16 / 5`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam: `0 / 0 / 0`
- exp226 control: fixed16 pseudo-cut geometry fold replay最大5、
  official OOF再生成0、GR correction/U-projection 0
- workers: 2

## 2026-07-26 実行承認

ユーザーの「実行してください」により、上記固定量のKaggle CPU Stage 0、
正規train Notebook採用、package、push、runを承認済みとした。
full OOF Stage 1、current-test、inference、submissionは承認範囲外で無効のまま。
Kaggle title上限内で意味を保持する48文字のcanonical slug/title
`exp406-loopclosed-multiwell-rgt-f16-stage0-train` /
`exp406 loopclosed multiwell rgt f16 stage0 train`を使う。

### push前package監査

- CPU / private / internet off / run-on-push: `true / true / true / true`
- kernel sources: exp226 / exp405 / exp065の3件
- embedded config SHA256:
  `1a996cae308f31f72040793b7d46802f3cb3b8310cad2e0336ce41601da60cda`
- push notebook SHA256:
  `162df3d0628bb899c17bc802baca8887a3f03db53884a1cb7fbb541ddac7b400`
- kernel metadata SHA256:
  `7cc8e4bb08ab4207602ede320c6a02394012a0c2541f8ea967d67402e19aebf1`
- local / packaged config SHA一致、id/title slug一致
- embedded実行量:
  1 diagnostic / 16 targets / 5 graph contexts /
  model config・booster・PF・HMM・Beam `0/0/0/0/0`
- full OOF / inference / submission authorization: `false / false / false`

## 2026-07-26 実装

ユーザーの「exp406を実装してください」でimplementation-onlyを承認後、
保存済みexp226 OOFがofficial suffix行しか持たずprefix512 controlを直接覆わない
矛盾を確認した。推奨案として提示したfixed16 pseudo-cut限定exp226 K16 geometry
replayについて追加承認を得た。

### 実装した内容

- 10章・3,171行のcompact self-contained train候補
- fail-closed compact self-contained inference候補
- exp405 decision status / decision SHA hard preflight
- exp226 OOF decompressed SHA / fold identity hard preflight
- exp386と同じround-robin `(fold, sorted well_id, offset)` fixed16 selector
- exp065 `native_overlap / threshold=1` Type-Well group SHA hard preflight
- outer-valid `MD/X/Y/Z/GR/TVT_input`限定readとdonor-fold exclusion
- H256/H128 block、±55/5 ft、same-Type-Well優先12 donors、top4 edge
- raw/rolling21/rolling101 robust NCCとSHA256固定circular control
- deterministic fundamental-cycle basisとHuber delta 5 ft / 10回IRLS
- edge/cycle/gauge logical SHAによるtarget-free freeze
- graph query / connected / finite gauge / cycle / negative control gate
- freeze後のfixed16-only exp226 original K16 field / Kappa geometry control
- prefix512 rolling-origin、target-free graph時間だけのStage 1 resource projection、
  Stage 0 diagnostic合計時間の別記録、全AND decision
- planned生成物、schema/logical/file SHA manifest

exp226 controlは各foldのouter-trainからraw/smoothed K16 field、near-strike
ANCC local-theta、adaptive Kappaを再構築するが、予測対象はfixed16 pseudo-cutだけ。
target ANCC、GR correction、U-projection、official OOF、full-well predictionは
生成しない。outer-train ANCCはtarget-free graph freeze後だけ読む。

### Notebook比較

- exp406 compact train: 3,171行 / 10章
- exp405 parent compact train: 2,756行 / 10章
- exp386 topology reference compact train: 2,718行 / 10章
- 同一exp helper import、`__file__`、薄い`main()` entrypointは不使用
- 既存の正規train/inference Notebookは明示採用前のため上書きしていない

### 静的・synthetic検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/*compact_selfcontained*.py \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/tests/test_exp406_loop_closed_multiwell_rgt_fixed16_stage0.py
.venv/bin/ruff check \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/*compact_selfcontained*.py \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/tests/test_exp406_loop_closed_multiwell_rgt_fixed16_stage0.py \
  --select F821,F811
.venv/bin/pytest -q \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/tests/test_exp406_loop_closed_multiwell_rgt_fixed16_stage0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/exp406_loop_closed_multiwell_rgt_fixed16_stage0_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/exp406_loop_closed_multiwell_rgt_fixed16_stage0_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/exp406_loop_closed_multiwell_rgt_fixed16_stage0_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp406_loop_closed_multiwell_rgt_fixed16_stage0/exp406_loop_closed_multiwell_rgt_fixed16_stage0_compact_selfcontained_inference.py
make validate-exp EXP=exp406_loop_closed_multiwell_rgt_fixed16_stage0
```

- py_compile: PASS
- Ruff F821/F811: PASS
- dedicated pytest: `13 passed`
- Jupytext train / inference round-trip: PASS
- strict experiment validation: PASS
- `make validate-template`: PASS
- 全体回帰`make test`: `1,178 passed / 7 skipped / 5 failed`
- Kaggle package / push / execution: なし
- local Notebook実行: なし

全体回帰の5件はexp406外の既存状態で、exp293 downstream contract本文と
historical hard-coded SHAの不一致2件、完了後configに対して実行前status/flagを
期待するexp296のstale test 2件、exp407で現configの`run_approved=false`に対して
旧`true`を期待するtest 1件だった。exp406専用12件は変更前の全体回帰内でもPASSし、
最終resource projection guard追加後の専用13件もPASSした。
他実験の履歴契約や実行承認状態は変更していない。

## コマンドログ

scaffold/designに加えてimplementation-onlyの静的・synthetic検証を実行した。

```bash
make new-steering EXP=exp406_loop_closed_multiwell_rgt_fixed16_stage0
make new-exp EXP=exp406_loop_closed_multiwell_rgt_fixed16_stage0 \
  SOURCE=templates/experiment
```

このimplementation-only検証時点ではKaggle、local Notebook実行、
正規Notebook採用は行っていなかった。後続の実行承認により正規trainだけ採用する。

実行承認後は次を行った。

```bash
make prepare-kaggle-notebooks \
  EXP=exp406_loop_closed_multiwell_rgt_fixed16_stage0 \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train \
  --title 'exp406 loopclosed multiwell rgt f16 stage0 train' \
  --run-on-push --strict"
make push-kaggle-train EXP=exp406_loop_closed_multiwell_rgt_fixed16_stage0
kaggle kernels pull \
  kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train -m
kaggle kernels status \
  kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train
kaggle kernels logs \
  kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train
kaggle kernels output \
  kentookumura/exp406-loopclosed-multiwell-rgt-f16-stage0-train
```

local Notebook実行は行っていない。

## 再現性メモ

- real graph: RNGなし
- circular control:
  `SHA256("exp406::circular::<well_id>::<block_id>")`
- parallel: 2 target wells、global RNGなし、immutable keyで再sort
- runtime: Kaggle CPU / GPUなし / internet off
- exp226 OOF decompressed SHA: `709eb7...c609`
- exp405 decision SHA:
  `e159cfb712a6ed81e78f4524febbf0d995375124a473a5056aad3c1347b648f0`
- edge logical SHA:
  `e6e75c23eda2c2ace4a899764fba08f2ba36c0144286f3314ab8fd79194b575a`
- cycle logical SHA:
  `e20bc07f67c67252d50c922e1cc353c532f2a1743c6467b77222af4cdc72b849`
- gauge logical SHA:
  `0b173575f627fbaff00e92447331eb01ec1469ba9df6246ed8cfa2bf000e688a`
- model / prediction / submission SHA: Stage 0では対象外
- deterministic anchor: false

## 禁止事項

- 承認範囲を超える正規inference Notebook採用・追加Kaggle run
- fixed16の選び直し
- exp386 threshold救済
- parameter grid、suffix truth/Formationの早期read
- unknown suffix prediction、scenario enumeration、ML/HMM/PF/Beam
- full OOF、current-test、inference、submission

## 次のアクション

exp406はparameter rescueなしで閉じる。full OOF/current-test/inference/submission、
fixed16再選択、donor/window/shift/edge/Huber/gate変更、prefix-only再実行は行わない。
Formation-derived exp386のroute棄却段階だけを分解する
`rgt_edge_cycle_path_rejection_readout`を低優先度P4の独立候補として残すが、
着手は別steering・別承認とする。
