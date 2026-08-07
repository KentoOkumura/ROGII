# exp335_signed_residual_meta_on_exp264 セッションノート

## 目的

exp264の既存74列selector compactを置き換えず、候補別signed residualの方向情報23列だけをstrict nested add-onlyするStage Sを実装し、実行前の承認境界を維持する。

## 現在の状態

- Route: `ml_model`
- 状態: Stage D固定tail guard FAILを保持したKaggle CPU inference version 3完了、submit-check PASS
- 親/control: corrected exp264 Stage C v6 / Stage D v3
- Selector CV: `8.430777428355306`（prior `10.974122864313635`、5/5 folds改善）/ downstream CV: `8.146107755881022` / Public LB: `7.517`
- 実装コード / Jupytext source / 採用対象notebook: あり
- Stage S model / nested compact / score: あり / downstream model・OOF: あり / CPU inference実装・test prediction・submission生成: あり / user-submitted ref `54928806`: Public LB `7.517`

## 変更点

- label: `true_tvt - candidate_tvt`
- selector objective: `regression_l2` 1 config
- selector surface: raw-test-safe 88特徴、12候補、2 legal domain、outer5-inner4
- new compact: candidate-specific 12 + existing-top1 annotation 8 + distribution 3 = 23列
- downstream: clean273 + saved74 + signed23 = 370特徴
- planned selector train: 20 CPU boosters
- planned downstream train: 15 GPU boosters
- selector/control retraining: 0 boosters
- hard selector、softmax/Viterbi、exp287 formation、objective/grid救済: 対象外

## コマンドログ

- 2026-07-21: `make new-steering EXP=exp335_signed_residual_meta_on_exp264`
- 2026-07-21: steeringの要件、設計、tasklistを記入し、23列schema、gate、再現性、承認境界を固定した。
- 2026-07-21: `make new-exp EXP=exp335_signed_residual_meta_on_exp264`
- 2026-07-21: テンプレートから実験scaffoldを作成し、設計文書とconfigだけを記入した。正規notebookは未実装scaffoldのまま維持した。

## 2026-07-21 Stage S実装

- ユーザーの「exp335を実装してください」をStage S実装の明示承認として受領した。既存設計の順序に従い、Kaggle package/push、preflight実行、20 CPU booster学習、Stage D、inference、submissionへは進めていない。
- `src/signed_residual_meta.py`を追加した。実装内容は、signed label formula、23列schema、exp264保存top-1 identity parity、全25 parent compact partitionのpre-fit SHA gate、fold 0先頭1,024行のparent key/top-1 alignment probe、exact chunk cursor、outer-train inner OOF / outer-valid 4-model平均、20-model LightGBM L2、metrics/gate、model/partition/reproducibility manifestである。
- parent exp264と同じcandidate順、88特徴、`max_train=60,000`、`max_valid=30,000`、`predict_chunk=20,000`、1,200 rounds / early stopping 80、LightGBM common config、sampling key `exp264/stage_c_*`、regressor seed offsetを固定した。変更はlabel/objectiveとsigned 23列だけである。
- unavailable/nonfinite candidateは暗黙の0 labelを作らずfail-closedとした。監査scoreにはactual signed residualを保存できるが、downstream partitionはkey/anchor + 23 signed特徴 + role metadataだけで、true TVT、actual label/error/rank/oracleを含めない。
- 正規`exp335_*_train.ipynb`は上書きせず、390行の`exp335_signed_residual_meta_on_exp264_compact_selfcontained_train.py`と16-cell / output 0の同名`.ipynb`を別名候補として作成した。parent train sourceは465行で、双方ともContents + 7 role sectionsを持つ。exp335はnotebook-safe runtime、承認/計算量、入力、preflight、実行、metrics、生成物をセルに展開し、1,246行の重いfold/Parquet/SHA helperを`src/`に残した。
- `config.yaml`は`stage_s_implemented_not_run`へ更新した。implementationだけtrue、`preflight_run_approved` / `selector_train_approved` / downstream / inference / submissionはfalse、control再学習0である。

### 実装検証

- `.venv/bin/python -m py_compile ...`: PASS
- `.venv/bin/ruff check ...`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...`: 16 cells、code 7 / markdown 9、output / execution count 0
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...`: PASS
- `rg -n "__file__|Path\(__file__\)" ...`: hit 0
- `.venv/bin/pytest -q tests/test_exp335_signed_residual_meta_on_exp264.py`: 6 passed
- `.venv/bin/pytest -q tests/test_exp264_candidate_selector_pipeline.py tests/test_exp335_signed_residual_meta_on_exp264.py`: 23 passed
- `make validate-exp EXP=exp335_signed_residual_meta_on_exp264`: strict PASS
- `make validate-template`: PASS
- `.venv/bin/pytest -q`: 512 passed / 2 skipped / 2 failed。失敗2件は既存`tests/test_exp296_exp223_self_gr_known_tvt_support_gate.py`がexp296の現行完了status `completed_train_side_guard_failed_closed`と`run_variant=false`に対し、旧`kaggle_cpu_*` / run-approved状態を期待する不一致であり、exp335変更箇所とは独立する。exp296は本タスクのscope外のため変更していない。

### 将来の予定（未承認・未実行）

```bash
make prepare-kaggle-notebooks EXP=exp335_signed_residual_meta_on_exp264 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp335-signed-residual-meta-on-exp264-train --title 'exp335 signed residual meta on exp264 train' --run-on-push --strict"
make push-kaggle-train EXP=exp335_signed_residual_meta_on_exp264
```

上記はKaggle package/push承認後だけ使用する。現時点では実行しない。

## 2026-07-21 0-booster preflight実行承認

- ユーザーの「実行してください」を、事前登録した段階的gateのうちKaggle package、Kaggle push、`signed_selector_preflight`実行の明示承認として受領した。
- 今回実行するstage: `signed_selector_preflight`
- 実行variant数: 0（学習なし）
- LightGBM config数: 0
- fold数: 0
- 合計booster数: 0
- 既存selector/control再学習: 0 boosters
- `selector_train_approved=false`、`run_selector_train=false`を維持する。preflight PASS後のouter5-inner4、1 objective、20 CPU booster学習は別承認であり、今回開始しない。
- 実行要求を、検証済みself-contained候補を正規`exp335_signed_residual_meta_on_exp264_train.ipynb`へ採用してKaggle packageを作る承認としても扱う。元の正規notebookは未実装scaffoldであり、仮説・特徴量・CVは変更しない。
- Jupytext sourceから正規train notebookを生成した。16 cells（code 7 / markdown 9）、output 0、execution count 0。`jupytext --test`、`py_compile`、`ruff`、exp264+exp335 contract test 23件、strict `validate-exp`はすべてPASSした。
- `prepare-kaggle-notebooks --notebook train --run-on-push --strict`でKaggle CPU packageを作成した。metadataはprivate、GPU=false、internet=false、competition source=`rogii-wellbore-geology-prediction`、kernel sources=exp263 cache train + exp264 selector train、run_on_push=trueである。
- 正規notebook SHA256: `3f28eb927f5222a0a76274478d7059cb692a9a7ac0c36534172a475c76ff89ef`
- bootstrap済みKaggle notebook SHA256: `55ea9b1b3539025196c8f63ab60ff62153e79e86cc93b5a665e19f11c748fd1`
- kernel metadata SHA256: `7de61d81dd254b83c4148223cee791e385b71b9bab23cb91f55e33e7b810273c`
- package内`src/signed_residual_meta.py` SHA256: `e87054913d64248dce5ec4d5d432a0b204eebd2131031766312bc1e3c09fe338`
- 同一slugの既存kernelは`kaggle kernels list --mine --search ...`で`Not found`。`kernels pull`の403は未作成kernelであることを一覧検索で確認した。
- Kaggle kernel version 1は0 boosterのまま約14秒でERROR。bootstrap 20 filesは成功したが、competition dataが`/kaggle/input/competitions/rogii-wellbore-geology-prediction`へmountされる環境で、resolverが`/kaggle/input/<slug>`と直下directoryしか見ていなかったため`competition train/test input root was not found`となった。学習処理、親SHA確認、model生成には到達していない。
- 既存exp264実行manifestで確認済みの`/kaggle/input/competitions/<slug>`を明示候補に加え、`sample_submission.csv`からの再帰fallbackも追加する。実験仮説、特徴量、split、学習設定、承認gateは変更しない。
- path resolver修正後もJupytext、compile、ruff、23 contract tests、strict `validate-exp`は全PASS。version 1のmetadataをpullし、private/CPU/internet off/2 kernel sources/competition sourceがversion 2 packageと一致することを確認した。
- version 2正規notebook SHA256: `755788de7f82a94afb3a91d7548247676d585a1af1e068cee1383252165e6dca`
- version 2 bootstrap済みnotebook SHA256: `092fc6e43c64c7d662d23cccd786546c23ef1d2086335c4f88220179dbe67e7d`

### Preflight結果

- Kaggle kernel: `kentookumura/exp335-signed-residual-meta-on-exp264-train` version 2、CPU、internet off、status `COMPLETE`
- 実行時間: notebook log上およそ75秒。preflight summary出力は約65秒時点
- technical preflight: PASS
- exp263 cache: manifest/catalog SHA一致、3,783,989 rows、773 wells、5 folds
- corrected Stage A: schema/catalog file SHAとlogical SHA一致、88特徴
- corrected Stage C: metrics/model manifest/compact manifest/schema SHA一致、74特徴、25/25 partition SHA確認、18,919,945 long rows
- candidate contract: 12候補、固定順一致、contract SHA `58950e002d4c388f02d8c98117c545aa6a600d6d4e3bbbed9277d23389551081`
- parent alignment probe: fold 0先頭1,024行、partition SHA `09e8dd0e4522997a9a4216ead377450439f52f9f959377fc2bf511df15337672`、top-1 value max abs error `0.0`、PASS
- signed compact: 23特徴、logical schema SHA `74abf31f057dfe29177221895e3e26c5a261e5b51defc04f081d6b140f2be44c`、file SHA `e57c6d5fde307eb26193a1c5efdb31accde94a6836df75de03100189d232021a`
- trained models: selector 0、control 0、downstream 0
- 保存物: `kaggle/output/preflight_v2/artifacts/stage_s_preflight.json`（SHA `55b01cd9306c0c3255a00ea55d7697778fc0ea7c0631b5635dfa16c6be537d15`）、`signed_compact_schema.json`（上記file SHA）、kernel log（SHA `2a3ab1955210dd98850efc5ac694169e3bcdb095bdb31f1c9bddd9492b98cc78`）

## 2026-07-21 Stage S selector学習承認

- preflight PASS後のユーザーの「実行してください」を、事前登録した`Stage S selector train`の明示承認として受領した。
- 実行stage: `signed_selector_train`
- active variant数: 1（`signed_residual_l2`）
- LightGBM config / objective数: 1（`regression_l2`）
- fold構成: outer 5 × inner 4
- 合計学習量: 20 CPU boosters
- 既存exp264 selector再学習: 0 boosters
- 保存済みexp264 control再学習: 0 boosters
- downstream Stage D: 0 GPU boosters（未承認）
- inference / submission: 未承認
- `selector_train_approved=true`、`run_selector_train=true`へ変更し、同じcanonical kernelへversion 3としてpushする。学習先はKaggle CPU、internet off、seed 42、`deterministic=true`、`force_col_wise=true`、`n_jobs=8`である。
- technical + score gateの判定後は必ず停止する。PASSしてもStage Dの15 GPU boostersは別承認なしに開始しない。FAIL時はobjective/grid救済を行わない。
- version 2をpullし、kernel id/title、private、CPU、internet off、2 parent kernel sourcesを照合した。version 3 packageのbootstrap内configも`stage=signed_selector_train`、selector承認/run=true、20 boosters、control/downstream=0であることを確認した。
- version 3 bootstrap済みnotebook SHA256: `37fbc0983bdf32199e3df47cb44c34f4a8e0fc72c5ef7521f9586e1e36ba4bd6`
- version 3 kernel metadata SHA256: `7de61d81dd254b83c4148223cee791e385b71b9bab23cb91f55e33e7b810273c`
- `make push-kaggle-train EXP=exp335_signed_residual_meta_on_exp264`でcanonical kernel version 3をpushし、`RUNNING`への遷移を確認した。
- ユーザーの「監視は止めていいです。完了したら連絡します。」に従い、Kaggle kernelは停止せず、こちらからのstatus/log pollingだけを終了した。完了metrics、gate、生成物はまだ未確認であり、結果として主張しない。

### 2026-07-22 Stage S完了確認

- ユーザーの完了連絡後にversion 3のlogs/statusを取得し、Kaggle status `COMPLETE`を確認した。実行時間はlog上約2,768秒（約46分）。
- 20/20 models、outer5-inner4の全組合せ、20 unique model SHAを確認した。best iterationは51–152、median 78.5。
- pooled signed-residual RMSE `8.430777428355306`、candidate別outer-train mean prior `10.974122864313635`、改善 `2.5433454359583294`。
- fold 0–4は順に`8.445401 / 8.556647 / 7.919418 / 8.429956 / 8.778735`、priorは`10.799173 / 10.961812 / 10.607191 / 10.953678 / 11.527623`で、5/5 foldsが改善した。
- technical gate: PASS。candidate順、88特徴、outer-valid除外、inner well-disjoint、outer-train inner OOF、outer-valid 4-model ensemble、25 partitions、18,919,945 compact rows、45,407,868 outer-valid long rowsを確認した。
- formula parity max abs error `0.0`、saved exp264 top-1 parity max abs error `0.0`。
- score gate: PASS（pooled改善 + 5/5 folds、要件4/5）。Stage S総合gate: PASS。
- 候補別では12候補中11候補が改善。`exp226_w500_50_50`だけprior比`-0.021853 ft`の小幅非改善だったため、Stage Dでは下流RMSE/tail guardで吸収可能かを確認する。
- small metrics/manifestsだけを`kaggle/output/stage_s_v3/artifacts/`へ取得した。model本体、45M-row score、19M-row compact parquetのローカル一括downloadは行っていない。
- metrics JSON SHA `7b3b51c2614528fc0448e6ecc4751db7e348b990023be00fba7c798c44bdafde`、model manifest SHA `2b626a0964d48da27b452e113afe8e05ee4342017fdd96bab80bea269e7390f5`、compact manifest SHA `237486930a0e6f7479d40d2b2d2ccb8e033e3787eb273c406d1eb5a3fc8a6b64`、outer-valid score SHA `631bd7c779f05c0594f3a0fc1c54df63d777b58bd691ca257aa34c8b80060a3c`、reproducibility manifest SHA `f4e37ce4bb8f38c2e9abc462b4625965612ac36815068cf9360766fff5d70ccb`、log SHA `d1a88d26bba4db50c9282f7e2a09e95dedd4ab6e31b42d9ed706b00cf1937e7f`。

## 再現性メモ

- seed policy: exp264 seed 42とfold/objective seed契約を固定し、新規RNGを追加しない
- stochastic components: 将来のCPU selector LightGBM、GPU downstream LightGBM
- PF/Beam regeneration: 0。保存済みexp264候補/compactをSHA固定入力として使う
- CPU/GPU runtime: preflightとStage S学習はKaggle CPU / internet off、Stage D version 2はKaggle T4 / internet offで完了
- input / feature schema SHA: exp263 manifest/catalog、candidate contract、exp264 Stage A schema/catalog、Stage C metrics/model/compact/schemaをすべて確認済み。新規23列schema logical/file SHAも生成・保存済み
- model manifest / model SHA: Stage S 20-modelとStage D 15-modelのmanifest・各model SHAを記録済み
- prediction / OOF SHA: outer-valid candidate score、25 signed compact partition、Stage D OOF SHAをmanifestで記録済み
- submission SHA: 対象外
- deterministic anchor: false
- rerun check: version 1はmount resolverでfail、version 2 preflight PASS、version 3 Stage S PASS。CPU modelのbitwise rerunは未実施のためdeterministic anchorとは扱わない

## 承認境界

- 今回承認済み・完了: backlog、steering、実験scaffold、config、設計記録、Stage S実装、Kaggle package/push、0-booster preflight、20 CPU selector boosters、Stage D実装、15 GPU boosters、固定gate判定
- 未承認: inference、submission
- control再学習: 禁止、0 boosters

## 次のアクション

1. exp335は固定tail guard FAILとしてクローズする。
2. gate緩和、signed residual objective/grid/threshold/特徴追加による同一実験の救済を行わない。
3. inference、submission、submission.csv生成を行わない。

### 2026-07-22 Stage D承認・push前実装監査

- ユーザーの「Stage Dに進んでください」を、事前登録済みのStage D実装と固定15 GPU booster実行の明示承認として受領した。
- active variant数: 1（`signed_residual_meta_addonly`）。LightGBM config数: 3（`lgb0/lgb1/lgb2`）。fold数: 5。合計: 15 GPU boosters。
- feature surface: clean273 + saved corrected Stage C v6 compact74 + Stage S v3 signed23 = 370列。
- saved exp264 Stage D v3 control再学習: 0 boosters。既存selector、PF/Beam/HMM、candidate生成の再学習/再生成: 0。
- runtime: Kaggle T4、internet off、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8。GPU bitwise deterministic anchorとはみなさない。
- input kernelはStage S v3、corrected exp264 Stage C v6、corrected exp264 Stage D v3、exp072 cache、exp145 learned likelihoodの5件。Stage C/S全25 partition、Stage S全20 model、saved control OOF/metricsをSHA検証してから1本目をfitする。
- scientific supportはpooled `>=0.03 ft`、4/5 nonworse folds、near/mid/1000+とhidden-like 2面非悪化、by-well delta p95 `<=0`、worst `<=+0.25 ft`、signed23 nonzero gainのAND gate。
- train-side promotionは上記に加え、clean273比worst-wellと+1/+3/+5 ft悪化well数がsaved exp264から増えないことを要求する。gateは緩和しない。
- inference、submission、submission.csv生成は今回のscope外。
- Jupytext convert/test、py_compile、ruff/F821、exp335/exp264/Kaggle notebook targeted 27 tests、strict experiment validation、template validationをPASSした。
- separate kernelは`kentookumura/exp335-signed-residual-meta-on-exp264-tvt-train` / title `exp335 signed residual meta on exp264 tvt train`へ固定した。push前pullは403で、同slugの既存kernelが未作成であることを確認した。
- 5 input kernelをpullし、Stage S id_no `128121640`、Stage C id_no `127485868`、saved Stage D id_no `127577193`、exp072 id_no `123000466`、exp145 id_no `124997617`を確認した。
- package metadataはprivate、T4、internet off、run-on-push、competition source 1、kernel source 5。bootstrap support fileは25件で、embedded config/source SHAと正規sourceが一致した。
- push前package SHA: config `04b18731474a32f5bcd66d005e438f186989b9239e84126af65b9ff627ee6da2`、`src/signed_residual_meta.py` `6968d28ade81ad233e80b279156f424806237231f45b38bdf76acf191f60f0da`、notebook `9d69b36f57e29e5dabb5fd5a699f95fccbc6a40acca2b4fa279243b83788dbad`、metadata `7086e3eb3c42d0f7061ad94cab4145a316b5daa20e1726de683f292ed62ed039`。
- `kaggle kernels push ... --accelerator NvidiaTeslaT4`でcanonical kernel version 1をpushし、`KernelWorkerStatus.RUNNING`を確認した。post-push pullはid_no `128232946`、private、T4、internet off、5 kernel sourcesを返した。
- remote notebookは19 cells、全cell sourceを正規packageと正規化比較して一致した。Kaggle側のserialization差によりfile byte SHAは異なるが、cell内容差は0。
- 重複実行防止のため、push直後にroot configの`run_downstream_train=false`へ戻した。実行中version 1のbootstrap config SHAは上記push前記録を正とする。

### 2026-07-22 Stage D version 1失敗調査・version 2修正

- canonical kernel version 1（id_no `128232946`）は`KernelWorkerStatus.ERROR`。traceback到達は約`519.7 sec`、LightGBM fitログとmodel生成はなく、実学習量は`0 / 15 GPU boosters`だった。
- Stage C/S全partition、Stage S全model、exp218 surface再構築までは通過した。失敗はsaved exp264 OOF truth parity直前で、`last_known_tvt`がrequired列とclean273 allowlist先頭の両方に含まれるまま列選択されたため、`base_frame["last_known_tvt"]`がshape `(3,783,989, 2)`になり、target `(3,783,989,)`との加算がbroadcast errorになった。
- 修正はretained base-column名の順序保持de-duplicationだけ。モデル入力`base_features`は従来のclean273を維持し、saved74 + signed23を含む最終370特徴、variant/config/fold、seed、gateは変更していない。
- `last_known_tvt`と`md_since`がbase featuresにも含まれるケースの回帰testを追加し、専用test `7 passed`、ruff F821/E9、py_compileをPASSした。
- version 1 log SHA: `1144525b138330c8a961e9c15da37b770ed72010216a3df9ec4c9a722e95d91f`。
- version 2も1 variant × 3 configs × 5 folds = 15 GPU boosters、saved control再学習0、inference/submission 0の同一契約で再実行する。
- version 2 packageはprivate / T4 / internet off / run-on-push、competition source 1件、kernel source 5件。package内`run_downstream_train=true`、root configは重複実行防止のため再び`false`とした。
- 修正後はexp264/exp335/Kaggle notebook targeted `28 passed`、Jupytext test、ruff F821/E9、py_compile、strict experiment/project validationをPASS。packageは19 cells、bootstrap 25 filesで、埋め込み`src/signed_residual_meta.py` SHAが正規sourceと一致した。
- version 2 push前package SHA: config `484f7d1a1a6631708e3f346ad504e20d3a4ad1b9faefe7aa9dc77904faaf2120`、`src/signed_residual_meta.py` `b5479fe5671fd999515973e2f08657ba8aeb80f85efe63e941d53dbb20c90d6b`、notebook `2b98f07ec63e7d56a3bac7f4bf33efa5be4a65b3afa9b97304ad70128cef0010`、metadata `7086e3eb3c42d0f7061ad94cab4145a316b5daa20e1726de683f292ed62ed039`。
- push前にcanonical version 1 metadataをpullし、id_no `128232946`、private、T4、internet off、competition sourceと5 kernel sourcesがversion 2 packageと一致することを確認した（source列順のみKaggle側で正規化）。
- `kaggle kernels push ... --accelerator NvidiaTeslaT4`で同じcanonical kernelへversion 2をpushし、`KernelWorkerStatus.RUNNING`を確認した。
- post-push pullはid_no `128232946`、private、T4、internet off、5 kernel sources。remote 19 cell sourceは正規packageと全一致し、埋め込み修正版SHA `b5479fe5...c90d6b`も一致した。
- ユーザーの既存指示どおりactive pollingは行わない。完了連絡後にstatus/logsを取得し、15 model実体、OOF/metrics/SHAと固定gateを評価する。

### 2026-07-23 Stage D version 2完了・固定gate判定

- ユーザーの完了連絡後、canonical kernel `kentookumura/exp335-signed-residual-meta-on-exp264-tvt-train` version 2 / id_no `128232946`のstatus `KernelWorkerStatus.COMPLETE`を確認した。
- 実行時間はlog最終eventで`20,017.035908815 sec`（約5時間33分37秒）。1 variant × 3 configs × 5 folds = 15/15 GPU boostersを完了し、saved control再学習は0だった。
- model manifestは15 unique `(fold, config)` slots、15 unique model SHA、全model 370特徴を記録した。best iterationは251--10,000、median 1,895。
- pooled OOF RMSEはsaved exp264 `8.460811237612477`から`8.146107755881022`へ`0.31470348173145446 ft`改善した。fold 1だけ`+0.375633 ft`悪化し、4/5 foldsは非悪化だった。
- 0--250、250--1000、1000+、hidden-like spatial、hidden-like typewell-purgedの全scopeがsaved exp264比で改善した。最大scope deltaは`-0.09041849893528164 ft`。
- by-wellは428/773 wellsが改善または同等、345 wellsが悪化。delta p95は`+1.7286570188927526 ft`で固定上限`<=0`をFAILした。
- worst well `fb03ae90`はsaved exp264比`+10.23875229975538 ft`で、固定上限`<=+0.25 ft`をFAILした。
- clean273比worst deltaはsaved exp264 `+14.482873080528407 ft`からexp335 `+17.77490971486976 ft`へ悪化。`+1/+3/+5 ft`悪化well数も`135/39/14`から`150/53/21`へ増えたためpromotion gateをFAILした。
- signed 23列のgainは`145410496278.61642`で非ゼロ、最大1特徴shareは`0.2529635458223868`。方向signalによる平均改善は確認できるが、tail safetyを満たさない。
- small artifactsだけを`kaggle/output/stage_d_v2/artifacts/`へ取得した。約106 MBのOOFとmodel payloadは一括downloadせず、reproducibility/model manifest内SHAを証拠とした。
- OOF SHA `8b28a3f29b981cbba118c9f98a5e7dd92e75613d87dddce39c2d162fb6a769b1`、metrics SHA `dd4502a5f8620820a023d7663b8335b20ea4e26ad847ff5c45757b6935c42ae1`、model manifest SHA `bfe917ba446096026c6e8bc6f0ac0a0a33c69b5d5602e140152caccc5d2d3bcd`、reproducibility manifest SHA `85ead119035604bd5559de566f57d0c5088e8f3c3cfbd4f6f5a13d8f07e21cba`、download済みkernel log SHA `48609a0899a2999a71aabeb9abbc51375009ebc3e6157073bb3068f57f6b3829`。
- local consistency auditでsmall artifact SHA、15 slots/models、370-feature contract、pooled/fold/scope/by-well/promotion count/signed gainの再計算一致をすべて確認した。
- scientific-support / promotionは`false / false`。事前登録どおりgateを緩和せず、signed residualのobjective/grid/threshold/特徴追加による救済、inference、submission、submission.csv生成なしでbranchをクローズする。

### 2026-07-23 CPU inference override・実装

- ユーザーの「推論に進んでほしいです。GPU quotaがないのでCPU実行にできますか。」を、固定tail guard FAILを保持した保存済みmodel inferenceの明示overrideとして受領した。
- runtimeはKaggle CPU、internet off、GPU false。学習variant/config/fold/boosterは0、control/selector/TVT model再学習も0。
- 推論時に使う保存modelはcorrected exp264 Stage C v6 parent selector 40本、exp335 Stage S v3 signed selector 20本、exp335 Stage D v2 TVT 15本。すべてmanifestと個別SHAをfit/predict前に確認する。
- raw current testからexp263 12候補、21 native-confidence列、88 selector特徴、exp218 380→clean273特徴を同じrun内で再生成する。保存済みpublic-test row artifactは入力に使わない。
- 各outerでparent 8 selectorからsaved74、signed 4 selectorからsigned23を生成し、同じouterのStage D 3 modelsへclean273 + saved74 + signed23 = 370列を渡す。
- GPU学習済みLightGBM text modelをCPU predictorで読むだけで、model変換や再fitは行わない。
- 親exp264 inferenceは1,272行 / 7 role sections、exp335 compact self-contained候補は1,448行 / 同じ7 role sections + signed生成、16 cells（code 7 / markdown 9）、output/execution count 0。薄いhelper呼び出しではなくcurrent-test生成、model/schema guard、推論、生成物保存をcellに展開した。
- `submission.csv`はsubmit-check用の推論成果物として生成するが、notebook/agentともcompetition submit APIを呼ばない。外部submissionは未承認。
- 正規inference notebookは未実装scaffoldだったため、検証済みcompact候補をユーザーの推論実行依頼に基づいて採用した。
- 専用contract testは25 passed。Ruff F821/F401/E9、py_compile、Jupytext `--test`、strict experiment/project validationをPASSした。ローカルnotebook実行はしていない。
- canonical packageは`kentookumura/exp335-signed-residual-meta-on-exp264-inference`、private、CPU、internet off、run-on-push。kernel sources 9件、private Stage C model dataset 1件、competition source 1件、bootstrap support 37 files。
- package内configは`run_inference=true`、`create_submission=true`、`submit_to_kaggle=false`、inference GPU falseを確認した。
- push前SHA: root config `60190383cf917568f1e1886f3bf926c02ec52456bfb065b8646c71f230f511ed`、compact source `e4139fda8e8da569083fb98a09d5f85c2a4d079a96dae3b465ae840a7737047e`、正規notebook `7cd759196f3fc61e360eee4571662135bd656019d784984e8183b4ec833f9841`、bootstrap notebook `4f89bb0071deca896f0c643c3a65d5a63383a512694f2e32c12e7a5ce355721c`、metadata `6f7a8e6501380719d18677507f43b58118a97579773e6e9e2e9be5a29b8327d1`。
- canonical slugのpre-push pullは403、owned kernel検索では同名inferenceはなく、T4 trainだけが見つかったため新規kernelとしてpushする。
- `make push-kaggle-infer EXP=exp335_signed_residual_meta_on_exp264`でcanonical kernel version 1をpushし、`KernelWorkerStatus.RUNNING`を確認した。
- post-push pullはid_no `128358534`、private、CPU（`enable_gpu=false` / `machine_shape=None`）、internet off、competition source 1件、kernel source 9件、dataset source 1件。remote 17-cell sourceはlocal bootstrap packageと全一致した。
- 重複実行防止のためroot configの`run_inference` / `create_submission`をfalseへ戻した。実行中version 1のembedded configはpush前SHAを正とする。

### 2026-07-23 CPU inference version 1失敗調査・version 2修正

- canonical CPU inference version 1は約381.9秒で`KernelWorkerStatus.ERROR`になった。
- 37 support filesのbootstrap、公開replay tracker compile、3 test wells / 14,151 rowsのbase feature生成、likelihood-PFを含むraw-test-safe feature再生成までは正常に完了した。
- 失敗は最初のmodel prediction前で、親推論から移植したchunk-size参照が`config["model"]["training"]`を要求した一方、exp335では同じ固定値を`config["model"]["selector"]["training"]`に保持していたため`KeyError: training`になった。CPU非互換、GPU quota、model/input SHA、feature schemaの失敗ではない。
- version 2では参照先を`model.selector.training.predict_base_row_chunk_size`へ修正するだけで、CPU/internet/GPU、入力source、40/20/15 saved models、88/74/23/273/370 feature契約、submission禁止は変更しない。
- version 2の専用contract testは25 passed。Ruff F821/F401/E9、py_compile、Jupytext `--test`、strict experiment/project validationをPASSした。
- version 2 package内は`run_inference=true`、`create_submission=true`、`submit_to_kaggle=false`で、修正済み設定参照を含む。push前SHA: config `f5b35f71428979b296609dc296076c1f183709975c8ad4758f2f59b8099c90fd`、compact source `7ed89640b16e129b875c7991494922e581c46242778d7b797f9fed8fa9f80a0d`、正規notebook `8bdb0fcd33bea45dd5b5100dff560312292efc5d7cf8991fddaa5ad37f15860e`、bootstrap notebook `055e805bf0e187098872481dd3416347c9b5cbe839f8d4018470745ea4163d0f`、metadata `6f7a8e6501380719d18677507f43b58118a97579773e6e9e2e9be5a29b8327d1`。
- `make push-kaggle-infer`でcanonical CPU kernel version 2をpushした。重複実行防止のためroot configの`run_inference` / `create_submission`は直後にfalseへ戻し、実行中packageのembedded configは上記push前SHAを正とする。
- version 2は約27.2秒で`KernelWorkerStatus.ERROR`になり、37 support filesのbootstrap後、feature生成前の固定承認ガードで停止した。進捗記録のため`inference.status`を変更したことが、notebookの期待する固定token `user_authorized_2026_07_23_cpu`と衝突した。
- version 3では`inference.status`を固定承認tokenへ戻し、進捗は`experiment.status`だけで管理する。version 1のchunk-size修正も維持し、model/input/feature/runtime/submit契約は変更しない。
- version 3 packageは固定承認token、修正済みchunk-size参照、`run_inference=true` / `create_submission=true` / `submit_to_kaggle=false`を同時に確認した。push前config SHA `1b55066753b87440659f2a0a1ec1afb0e5e03d78e10df46215201ac6780c4512`、bootstrap notebook SHA `af319cf342ab84d3f1d90458c423d53b7a24df859ca2d9063d9cb6318bee9d4b`。
- canonical CPU kernel version 3をpushし、重複実行防止のためroot configのrun/create flagsを直後にfalseへ戻した。
- version 3が`RUNNING`の間に、ユーザー指示に従ってactive pollingを停止した。完了連絡後にstatus/log/outputを取得し、submit-checkを行う。

### 2026-07-23 CPU inference version 3完了・submit-check

- ユーザーの完了連絡後、canonical kernel version 3 / id_no `128358534`の`KernelWorkerStatus.COMPLETE`を確認した。Kaggle metadataはprivate、CPU（`enable_gpu=false` / `machine_shape=None`）、internet off、competition source 1、kernel source 9、private dataset source 1だった。
- notebook内推論runtimeは`387.808 sec`、生成物表示は約`407.98 sec`、Kaggle log最終eventは約`419.49 sec`。14,151 rows / 3 wellsを処理し、学習boosterは0だった。
- raw current testから12候補、21 confidence列、88 selector特徴、source380→clean273を再生成した。parent compact 74、signed compact 23、final 370特徴の順序をouter別に検証した。
- parent selector 40、signed selector 20、TVT model 15は全slot使用、各群のmodel SHAは40/40、20/20、15/15で一意だった。formula parityとsigned top-1 value parityの最大絶対誤差はいずれも`0.0`。
- `submission.csv`は14,151 rows、列`id,tvt`、sampleとheader・行数・ID内容/順序が完全一致した。ID重複、NaN、Infはなく、skill checkerとrepo checkerはいずれもPASS、WARN/FAIL 0だった。
- prediction統計はmin `11591.086914`、max `12239.015625`、mean `11905.253906`、std `278.593842`。submission SHAは`9d163b11fbea5c6a1e807f9681aaf39916bb5682e35ea874d3acd981f922a14f`、decompressed prediction SHAは`67af935f64020555fdf611975b1ffff874f233d64e43c0095a8caa8aef292da0`。
- small metrics/schema/missingness/logだけを`kaggle/output/inference_v3/`へ保存した。metrics/reproducibility SHAは`4fe5402aa0c4737a3edda8de39a045ee60faf6ecb71d968a428c04c8426f0c81`、downloaded log SHAは`0e35fdff77d3145a7b06dfac3ec15b1b437d2a4d8735a0fbebcb0aab4e888153`。submissionと行単位predictionは実験ディレクトリへ常設していない。
- Stage D scientific-support / promotion FAILは`false / false`のまま保持した。外部competition submissionは未承認かつ未実行。

### 2026-07-24 code submission scoring完了

- ユーザーからscoring完了の連絡を受け、Kaggle submissionsとmonitor scriptを確認した。
- submission ref `54928806`、submitted UTC `2026-07-23 13:23:43.643000`、status `COMPLETE`、Public LB `7.517`、Private LB未公開。descriptionは空欄だが、このスレッド直後の最新code submissionであり、ユーザー確認済みexp335として記録する。
- monitorは完了後に開始したため表示上のrun-timeは0分であり、実際のscoring所要時間として扱わない。monitor log SHAは`c63c05581c8b1bf53383ce1c8e3f5507bb6f607a48feefc70d048692016790be`。
- Public LBは直前のML anchor exp287 `7.530`を`0.013`、親exp264 `7.562`を`0.045`、保存済みensemble anchor exp082 `7.601`を`0.084`改善し、追跡中のPublic-LB reference anchorを更新した。
- ただしStage Dのby-well tail guard FAILとtrain-side非promote判断は維持する。後付けのgate緩和、objective/grid/threshold/特徴救済は行わない。
