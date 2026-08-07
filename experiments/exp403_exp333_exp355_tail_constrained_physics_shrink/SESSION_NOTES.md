# exp403 セッションノート

## 目的

exp263の2つの物理成分を、保存済みexp333 / exp355へ同時置換した平均gainを、
outer-trainだけで選ぶtail制約付きscalar shrinkで安全に保持できるか検証する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle train version 4 COMPLETE・scientific FAIL・terminal close
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- CV / LB: `8.238331667 / なし`
- steering:
  `.steering/20260726-exp403-exp333-exp355-tail-constrained-physics-shrink/`
- 正規train Notebook: compact self-contained候補を採用済み
- `settings.py`: 未編集
- compact self-contained Jupytext train/inference候補: 実装済み
- 専用contract test: `12 passed`
- Kaggle package / run: version 4完走、実行承認は消費済み
- inference / submission: 未実施、FAILにより閉鎖

## 2026-07-26 実装

ユーザーの`exp403を実装してください`という指示を実装承認として、凍結済み設計の
範囲だけをコード化した。正規Notebookの上書き、Kaggle package / push / run、
inference有効化、submissionは承認範囲に含めていない。

### 実装したもの

- 別名compact self-contained train Jupytext source / Notebook候補
- fail-closed inference Jupytext source / Notebook候補
- exp263 manifest / partition file SHA、exp333 / exp355 raw・decompressed SHA guard
- exp263 generation fold単位の3 primitive streaming load
- `well_id,row_idx`のuint64 global keyによるexp333 / exp355 target-free join
- exp226 parity、exp263 float32/float64 formula parity、full置換、correction freeze
- 2 fold ledgerのwell内一定性、support、631/773 mismatch、cross-tab
- freeze後だけraw suffix truth / exp115 hidden-like roleを読むaccess ledger
- 固定9 λのouter-train eligibility、最大positive値、zero fallback
- pooled / fold / distance / hidden-like / by-well / persistent-offset / recovery gate
- cross-fit OOF、λ ledger、fold ledger、freeze / promotion / SHA manifestの保存
- initial synthetic contract test 10件、実行時guard追加後の最終test 12件

### Notebook構成

親exp263にはcompact self-contained train候補がなく、正規train sourceは335行の
orchestration中心構成だった。exp403候補は約2,000行、10章で、runtime/config、
入力/SHA、source freeze、truth-late、λ選択、metrics、生成物保存までNotebook上で
追える。`__file__`は使用していない。

### 実行ガード

- `implementation.enabled=true`
- `execution.kaggle_train_run_approved=false`
- `execution.run_train=false`
- `implementation.training_enabled=false`
- inference / submission flagはすべてfalse

したがってNotebook候補を誤って開いてもreadoutは開始しない。runには別承認と
config更新が必要である。

## 2026-07-26 設計確定

### 根拠

保存済みOOFのread-only診断では、固定重みの両置換はexp263再構成
`8.238331745`から`8.159425494`へ`0.078906251 ft`改善した。
一方、改善foldは3/5、by-well p95は`+1.983209 ft`、worst
`86454a6f +13.412007 ft`だった。平均signalとtail failureを分離して
検証するため、full置換のweightを変えずexp263へscalar shrinkする。

### 固定式

```text
exp263 =
    0.50*exp226_k16
  + 0.25*likpf_mean
  + 0.25*exp209_exact_hmm

full =
    0.50*exp333_stage1
  + 0.25*likpf_mean
  + 0.25*exp355_stage1

candidate = exp263 + lambda_fold*(full-exp263)
```

λ候補は
`0, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1`。
outer-trainでpooled gain、near/mid/1000+、by-well p95、worstの全条件を
満たす最大positive λを選ぶ。なければ0へfail closedする。

### Fold契約

- reporting / λ calibration: exp226 outer fold
- exp263 cache partition: exp263 generation fold
- 両者は独立したgroup-safe ledger
- 既知のfold label mismatch: 631 / 773 wells
- global join: `well_id,row_idx`
- fold equality assertは禁止し、両ledgerのsupport / well内一意性 /
  cross-tabを保存する。

### 実行量

設計上:

- scientific policy: 1
- calibration λ: 9
- reporting folds: 5
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam well-runs: `0 / 0 / 0`
- parent/control rerun: 0
- CPU / internet off

このdesign-only時点ではすべて未実行。

### Promotion

- positive λ folds `>=4/5`
- pooled gain vs exp263 `>=0.03 ft`
- improved folds `>=4/5`
- near / mid / 1000+ / hidden-like 2面 delta `<=+0.02 ft`
- by-well p95 `<=0`
- worst `<=+0.25 ft`
- persistent episode非増加
- 512-row recovery非悪化

FAIL時はλ、component weight、候補集合、gate、routerを救済せず閉じる。

## 入力証拠

- exp263 cache manifest:
  `85e60ac10b50197fa44ea29faffcbba81bd0746114bc53bae0f5cc537a26bb9e`
- exp333 OOF raw / decompressed:
  `70b623d4...2dc` /
  `f2ebc6f6...e5a`
- exp355 OOF raw / decompressed:
  `28da6ffb...a41` /
  `3c49f25e...de3`
- exp355 prediction logical SHA:
  `634303f0...e21`
- hidden-like assignment:
  `5f9ac9fa...597`

## コマンドログ

設計時に実行したコマンド:

```bash
make new-steering EXP=exp403_exp333_exp355_tail_constrained_physics_shrink
make new-exp EXP=exp403_exp333_exp355_tail_constrained_physics_shrink \
  SOURCE=templates/experiment
make update-summary
make validate-exp EXP=exp403_exp333_exp355_tail_constrained_physics_shrink
.venv/bin/python \
  .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py \
  exp403_exp333_exp355_tail_constrained_physics_shrink --root .
```

- `make validate-exp`: strict PASS
- design assertion: route / status / implementation flag / λ候補 /
  fold mismatch / execution countを確認してPASS
- document review: core evidence、比較軸、未実装状態を確認

学習、推論、Kaggleコマンドは実行していない。

実装時に実行した検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp403_exp333_exp355_tail_constrained_physics_shrink/*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp403_exp333_exp355_tail_constrained_physics_shrink/*compact_selfcontained*.py \
  --select F821,F811
.venv/bin/pytest -q \
  tests/test_exp403_exp333_exp355_tail_constrained_physics_shrink.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp403_exp333_exp355_tail_constrained_physics_shrink/*compact_selfcontained*.py
```

- `py_compile`: PASS
- Ruff F821/F811: PASS
- 専用test: `10 passed`
- Jupytext変換: train / inferenceともPASS
- Jupytext `--test`: train / inferenceともPASS
- strict `make validate-exp`: PASS
- Kaggle run / raw truthを使うlocal smoke: 実施していない

## 再現性メモ

- seed policy: RNGなし
- stochastic components: なし
- PF/Beam/HMM/ML: 保存済み予測load-only、再実行0
- execution: single-process streamingを予定
- gzip evidence: decompressed content SHA
- Parquet evidence: manifest / partition logical content SHA
- truth: source prediction、formula、identity、content SHA freeze後だけrawからjoin
- model / prediction / submission SHA: 現時点では非該当
- Kaggle kernel id / version: 未作成
- deterministic anchor: false

## 禁止事項

- outer-valid / Public LBによるλ選択
- result後のλ grid、weight、component、scope、gate変更
- per-well / per-row router
- exp333 / exp355 / exp263 control再実行
- exp263 foldとexp226 foldの同一性assert
- promotion前のinference / submission

## 2026-07-26 Kaggle train実行承認

ユーザーの「実行してください」を、次の範囲に限った明示承認として記録した。

- compact self-contained train候補の正規train Notebookへの採用
- Kaggle package作成
- CPU / internet offのKaggle train readout 1 run
- 保存済みOOFだけを読む科学方策: 1
- 固定calibration λ: 9
- reporting fold: 5
- model config / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam well-run: `0 / 0 / 0`
- parent / control再学習: 0

推論実装・inference run・submission生成・competition submitは承認範囲外とする。
promotion PASS後も自動実行しない。

Kaggle train slugは、長さ58の旧案を避け、両置換元を識別できる43文字の
`kentookumura/exp403-exp333-355-tail-physics-shrink-train` に固定した。
push前の同名kernel pullは403で、既存Notebookは確認できなかった。

正規train Notebook採用後の再検証:

- canonical train: 22 cells（code 10 / markdown 12）
- Jupytext `--test`: PASS
- `py_compile`: PASS
- Ruff F821/F811: PASS
- 専用test: `10 passed`
- strict `make validate-exp`: PASS

Kaggle package検証:

- private / CPU / internet off / run-on-push: PASS
- competition source: 1
- kernel source: exp263 / exp333 / exp355 / exp115 の4件、順序も一致
- packaged notebook: bootstrap 1 cell + canonical 22 cells
- canonical notebook body一致: PASS
- loose config / package config / bootstrap embedded config byte一致: PASS
- initial embedded config SHA256:
  `adc2a6d5a5e483a43f298f2b9637df9a9783f9686b9a693a8815a6d6dcf24d5f`

package作成済み状態をconfigへ記録したため、push直前にpackageを再生成し、
最終embedded config SHAとbyte一致を再確認する。

## 次のアクション

packageを再生成・再検証してKaggleへpushし、完了まで監視する。
promotion PASS後のcurrent-test inferenceは別承認とする。

## Kaggle train v1 ERROR

- kernel/version/id_no:
  `kentookumura/exp403-exp333-355-tail-physics-shrink-train / 1 / 128628482`
- push: `2026-07-26 00:13:43 UTC`
- status: `ERROR`
- error time: `32.222 sec`
- scientific stage: branch source読込中、truth読込前、λ選択前
- error: `exp333 upstream prediction content SHA mismatch`

exp333 gzipのraw SHAとdecompressed SHAは先に一致しており、artifact違いや破損では
ない。失敗したのは、exp333 producer実行時の
`pandas.hash_pandas_object`をCSV再読込後の別dtype / pandas環境で再計算する
非portableな検査だった。

技術修正:

- exact raw gzip SHAとdecompressed SHAを引き続き必須guardにする。
- exp333 producer logical SHAはprovenanceとして保持するが、別環境での一致assertには
  使わない。
- exp333 / exp355の選択予測surfaceを、key sort・整数正規化・IEEE float hexによる
  `keyed_float_hex_v1` SHAとしてtruth読込前に新規freezeする。
- scientific policy、λ、fold、gate、入力artifactは変更しない。
- 追加のmodel / PF / HMM / Beam / parent rerunは0。

これは同じ承認済みscientific runを完了させるための実行環境修正としてversion 2へ
再pushする。

## Kaggle train v2 ERROR

- kernel/version/id_no:
  `kentookumura/exp403-exp333-355-tail-physics-shrink-train / 2 / 128628482`
- push: `2026-07-26 00:19:14 UTC`
- status: `ERROR`
- error time: `58.078 sec`
- scientific stage: exp263 generation fold 0のsource式assembly中、truth読込前、
  λ選択前
- error: exp263 float64/float32 formula parity
  `0.0009765625 ft`

exp263 componentはfloat32で保存され、exp403ではmetric計算のためfloat64へ正確に
昇格して固定式を再構成する。保存時と同じfloat32演算結果との差
`0.0009765625 ft`は、このTVT magnitudeにおけるfloat32 1 ULPであり、
component/formula不一致ではない。

技術修正:

- 既存absolute tolerance `1e-5 ft`は維持する。
- absolute toleranceを超えてもfloat32換算で最大1 ULP以内なら同値とする。
- generation partitionごとにabsolute最大差とfloat32 ULP最大差を保存する。
- scientific policy、component weights、λ、fold、promotion gateは変更しない。

同じ承認済みscientific runのtechnical rerunとしてversion 3へ再pushする。

## Kaggle train v3 ERROR

- kernel/version/id_no:
  `kentookumura/exp403-exp333-355-tail-physics-shrink-train / 3 / 128628482`
- push: `2026-07-26 00:24:38 UTC`
- status: `ERROR`
- error time: `116.819 sec`
- source freeze: 全5 generation partitions完了
- truth access before freeze: 0
- truth rows attached: 0
- λ選択 / scientific metrics: 未到達
- error: horizontal CSVの`TVT` usecols不一致

Kaggle inputには同じwell名を持つ複数directoryがあり、旧resolverは最初に見つけた
horizontal CSVの親directoryをschema確認なしで採用していた。その結果、公式train
より先に`TVT`を含まないtarget-free duplicateを選択した。

技術修正:

- well identityだけでなく、先頭3 wellsすべてで`TVT`と`TVT_input`のheaderを
  要求する。
- 条件を満たす候補ではcompetition sourceの`.../rogii-wellbore-geology-prediction/train`
  を優先する。
- schema不完全なduplicateをskipする専用testを追加する。
- truth-late順序、科学設定、λ、gateは変更しない。

同じ承認済みscientific runのtechnical rerunとしてversion 4へ再pushする。

## Kaggle train v4 COMPLETE / scientific FAIL

- kernel/version/id_no:
  `kentookumura/exp403-exp333-355-tail-physics-shrink-train / 4 / 128628482`
- push: `2026-07-26 00:29:53 UTC`
- status: `COMPLETE`
- scientific runtime / final log: `172.418 / 327.001 sec`
- peak RSS: `1.921 GB`
- rows / wells: `3,783,989 / 773`
- technical gate: PASS
- scientific promotion: FAIL
- decision: `close_without_lambda_component_weight_gate_or_router_rescue`

Technical evidence:

- input SHA、row/well、fold support、finite、runtime、RSS、execution count: 全PASS
- reporting/generation fold label mismatch: expectedどおり`631 / 773 wells`
- truth columns read before freeze: 0
- full truth rows late-attached: `3,783,989`
- official raw truth:
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train`
- control reference差: `7.8781e-08 ft`
- full replacement reference差: 0
- exp263 formula float32 parity: 全5 partitions最大`0.0009765625 ft / 1 ULP`

Scientific result:

- rebuilt exp263 control: `8.238331667`
- full replacement reference: `8.159425494`
- cross-fit candidate: `8.238331667`
- λ fold: `[0, 0, 0, 0, 0]`
- current-test λ: 0
- positive λ folds / improved folds: `0 / 5`, `0 / 5`
- pooled gain: 0
- fold RMSE:
  `7.233137 / 8.251973 / 8.660235 / 8.364633 / 8.581319`

最小positive λ`1/64`でも、outer-train gainはfold順に
`0.005785 / 0.007293 / 0.007314 / 0.007919 / 0.007752 ft`で固定下限
`0.01 ft`に届かなかった。by-well delta p95も
`+0.023577 / +0.023577 / +0.023597 / +0.026743 / +0.026743 ft`で
`<=0`を全foldで破った。positive eligible countは全fold 0だった。

SHA:

- source content:
  `6c2ec0157d3a397992fdaa9678c6ef79de63fba58aa4ed85c7a3682f95b41cbd`
- prediction raw / decompressed / content:
  `dfbf21ceba7aab7713b835257045666651db467bf29d38055b709944a3aad350` /
  `7b49a8c2ff392ffbea03ad20ee35d5e277ca77b5cdcd190d1adc16021da27b35` /
  `17a7bae695c56500e55d1b724b7ca29908d70b4a2cb39e03d618016f02d1618a`
- gate / summary:
  `094c59ee5f6dbd1335332c950c00cab68bf84449f67403693ec9848c2d838592` /
  `d8a79932fbfe2fe18471e805a36d1f1da963bd8a3069532a464f5e96b59f372b`

ログに加え、lambda/fold/scope/by-well/gate/freeze/SHAの小容量artifactだけを
`--file-pattern`で取得して検証した。128 MB predictionやfrozen Parquetを含む
output archive全体は取得していない。

full-run承認は消費済みとしてtrain/package run承認と`run_train`をfalseへ戻した。
inference / submissionは実施せず、same-OOF rescueは禁止のままterminal closeする。
