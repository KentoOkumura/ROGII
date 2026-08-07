# exp287_fold_safe_formation_74_addonly_on_exp264 セッションノート

## 目的

exp218監査でfull-train formation reference依存とされたbase-replay 74列をfold-safeに再生成し、
修正版exp264の347列へadd-onlyした421列variantを、保存済みexp264 OOFと比較できる状態にする。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle T4 version 5完了・promotion guard FAIL・branch closed
- 親: corrected exp264 Stage C v6 / Stage D v3
- active variant: 1
- LightGBM config / fold / booster: 3 / 5 / 15 GPU
- parent/control retraining: 0
- execution stage: `fold_safe_formation_addonly_train`
- completed / planned GPU boosters: 15 / 15
- CV / parent / delta: 8.136708 / 8.460811 / -0.324103 ft
- `execution.run_approved`: false
- inference / submission: disabled / disabled

## 実装内容

- `.steering/20260719-exp287-fold-safe-formation-74-addonly-on-exp264/`を作成した。
- audit CSVの`status=fail`かつ`family=base_replay`かつ
  `dependency=full_train_formation_reference`に一致する74列をSHA・列順込みで固定した。
- `src/fold_safe_formation_pipeline.py`へ次を実装した。
  - raw train/current-test schema audit
  - FormationPlaneKNN / DenseANCCImputerのfold-local fit
  - outer-train targetのself-exclusion、outer-validのouter-train-only reference
  - current-test targetがformation列を読まないfeature contract
  - 5 fold × train/valid feature cache、logical content SHA、Parquet SHA
  - 既存347列とのfull-row exact duplicate、固定sample Pearson/Spearman report-only audit
  - saved exp264 corrected OOFのSHA / RMSE / row alignment guard
  - 421列variantの15 GPU booster学習、model manifest / OOF / importance / metrics保存
  - pooled / fold / near / mid / 1000+ / hidden-like / by-well promotion guard
- canonical train notebookは7章、351行。親exp264 train notebookは同じ7章、465行。
  親にcompact self-contained別版はなく、重いfold generation / LightGBM本体を`src/`へ残しつつ、
  notebook上でcost、入力、固定74列、reference契約、実行分岐、metrics、SHAを追える構成にした。
- inference notebookはpromotion前のfail-closed gateとして実装した。

## コマンドログ

- `make new-steering EXP=exp287_fold_safe_formation_74_addonly_on_exp264`: steering作成。
- `make new-exp EXP=exp287_fold_safe_formation_74_addonly_on_exp264`: scaffold作成。
- `py_compile` / `ruff --select F821,F401,E9`: helperとnotebook sourceでPASS。
- `pytest -q tests/test_fold_safe_formation_pipeline.py`: 5 passed。
- `pytest -q`: 全repo test PASS。
- train / inference Jupytext conversion: 正規`.ipynb`へ変換。
- train / inference `jupytext --to ipynb --test`: PASS。
- `make validate-template`: PASS。
- `make validate-exp EXP=exp287_fold_safe_formation_74_addonly_on_exp264`: strict PASS。
- `review_exp_docs.py exp287 --root .`: core evidence categoriesあり。
- 初期実装時点ではローカルnotebook実行、Kaggle prepare/push/run、model fit、inference、submissionは未実行だった。

## 固定scientific contract

- formation features: 74、exp111系27列 / dependent GRWR 6列を含めない。
- base surface: clean 273、nested compact 74、final 421。
- reference: outer-train固定。train targetはself-exclude、valid targetはouter-train-only。
- current-test: all-train reference、target formation columns read=false。
- duplicate/correlation: report-only、prune=0。
- training: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control=0 boosters。
- guard: pooled `<=-0.02`、4/5 folds、全scope `<=+0.02`、worst `<=+0.25`、
  +1/+3/+5 ft counts nonincrease。

## 再現性メモ

- formation generatorはRNGなし。wellをsortし、query workers=1、immutable idで再整列する。
- GPU LightGBMは`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、threads 8。
- feature cacheはfold-roleごとにfile SHAとlogical float32 content SHAを保存する。
- Stage C manifest/schema/partition、clean allowlist、availability audit、saved exp264 OOF、hidden-like assignmentを
  SHA固定する。
- model manifest / OOF / metrics / feature importance / by-wellのSHAを実行時に保存する。
- GPU bitwise同一は未確認なのでdeterministic anchorとは扱わない。

## 次のアクション

1. fixed promotion guard FAILとしてbranchを閉じる。
2. guard緩和や同一OOFでのfeature/grid/threshold救済は行わない。
3. current-test生成、inference、submissionは行わない。
4. exp276再検証でfold-stableなtarget-free risk familyが得られた場合だけ、別承認で0-boosterのformation tail attribution readoutを検討する。

## 2026-07-19 Kaggle T4 train v1実行承認

ユーザーの「実行してください」を、exp276を待たない次の限定scopeへの明示承認として記録した。

- active variant: 1 (`fold_safe_formation_74_addonly`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計: 15 GPU boosters
- saved corrected exp264 control再学習: 0
- approved scope: `fold_safe_formation_74_addonly_1_variant_3_lgbm_configs_5_folds_15_gpu_boosters_no_control_retraining_train_v1_only`
- accelerator: Kaggle `NvidiaTeslaT4`
- inference / submission: disabled / disabled

exp276 corrected revalidationの先行確認は優先順の提案であり、実装・train pushのhard prerequisiteではない。
今回の明示指示により先行条件を解除した。承認はtrain v1だけに適用し、guard通過後の
current-test生成、inference、competition submitは別判断とする。

### push前package監査

- `task`コマンドは環境に存在しなかったため、同等の`make` targetへ切り替えた。
- `make validate-exp EXP=exp287_fold_safe_formation_74_addonly_on_exp264`: strict PASS。
- `make validate-template`: PASS。
- dedicated tests: 5 passed。py_compile / ruff / train・inference Jupytext testもPASS。
- 再計算した実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0。
- 初回full-name id/title `kentookumura/exp287-fold-safe-formation-74-addonly-on-exp264-train` /
  `exp287 fold safe formation 74 addonly on exp264 train`はslug一致を確認したが、53文字で
  `SaveKernel 400`となり、Kaggle runは開始されなかった。
- 意味を保った43文字のcanonical id/title
  `kentookumura/exp287-foldsafe-form74-addonly-exp264-train` /
  `exp287 foldsafe form74 addonly exp264 train`へ同じexp287内で揃えて再prepareした。
- metadata: private / T4 / internet off / `run_on_push=true` / competition source 1件 / kernel sources 4件。
- bootstrap support zip: 22 files、SHA256 `d309509b9e3b765b10b25d664ea63e7ba469061abac3ebcd2eec0af6347830d2`。
- 埋め込みconfig SHA256: `2f77d8959f16a4cbc1673bf3cf6462d86dcc3a65d9cf4205736d3ef06a6459d8`。
- 埋め込みconfigで`run_approved=true`、train stage、15 boosters、control再学習なし、
  inference / submission無効を確認した。

### Kaggle train v1 push

- 短縮canonical slugで`kaggle kernels push ... --accelerator NvidiaTeslaT4`: version 1成功。
- URL: `https://www.kaggle.com/code/kentookumura/exp287-foldsafe-form74-addonly-exp264-train`
- push後pull: PASS、Kaggle kernel `id_no=127856426`。
- Kaggle側metadata: private / `NvidiaTeslaT4` / internet off / competition source 1件 / kernel sources 4件。
- version 1は約12秒でERROR。15 boostersの学習には到達していない。
- 原因: competition sourceは
  `/kaggle/input/competitions/rogii-wellbore-geology-prediction`へmountされたが、
  `find_competition_input_root()`が`/kaggle/input`直下だけを探索し候補0件になった。
- 対応: competition slugのnested/direct明示パスを優先し、2階層限定fallbackを持つresolverへ修正する。
- 同じcanonical kernelへversion 2として再pushし、slugや実験番号は増やさない。

### version 2 push前監査

- train Jupytext変換・test、py_compile、ruff、strict exp validation、dedicated tests 5件: PASS。
- package metadata: version 1と同じcanonical id/title、private / T4 / internet off / `run_on_push=true`。
- notebook本体とsupport zip内train sourceの両方にnested competition path resolverを確認した。
- support zip SHA256: `931f524280f2acedaf822f971572e9eff1df268da263a15ad2e9c9bef6f73905`。
- 埋め込みconfig SHA256: `5fa2b415d03302678799aecac4c4d8d433dc3053688cea4e27802eccf04db731`。
- 実行量は変更なし: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0。
- 同じcanonical kernelへversion 2 push成功。初回statusは`RUNNING`。

### 監視停止

- 2026-07-19 16:42 JSTまでversion 2の`RUNNING`を確認した。
- ユーザー指示によりローカルのstatus pollingだけを停止した。
- Kaggle kernel version 2自体は停止・cancelしていない。
- ユーザーから完了連絡を受けた後、同一kernelのlogsと必要なOOF / metrics / SHA生成物を確認する。

## 2026-07-19 Kaggle train version 2失敗診断

- ユーザーの失敗連絡後、同一canonical kernelのstatus / logsを取得し、version 2が
  `KernelWorkerStatus.ERROR`と確定した。
- runtimeは475.305秒。raw schema auditとexp218 surface生成には進んだが、
  `load_saved_exp264_control()`のSHA gateで停止し、LightGBM fitは0 / 15 boostersだった。
- 観測SHAは`b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`、
  config期待SHAは`7367983f3053186e0a6adf18c0f145302b0451332625fb679357f3c1326dafee`。
- exp264正規記録を照合すると、`b11c5005...9ae2`が修正版Stage D v3（347列、RMSE 8.460811）のOOF、
  `7367983f...dafee`はfeature-availability leakageで無効化された旧Stage D v2（RMSE 7.805644）のOOFだった。
  Parquetの再圧縮差ではなく、scientific parent versionの取り違えである。
- exp287は親RMSEだけv3、OOF SHAだけv2を固定していた。SHAを`b11c5005...9ae2`へ修正し、
  exp264 `corrected_stage_d` metricsのkernel version / OOF SHA / RMSEと照合する回帰testを追加した。
- 誤設定を含むversion 1承認scopeは消化済みとして`execution.run_approved=false`へ戻した。
  version 3は同じ1 variant / 3 configs / 5 folds / 15 GPU boosters / control再学習0だが、再push前に明示承認を得る。
- 修正後の待機packageは同じcanonical id/title、T4、internet off、`run_on_push=false`で生成した。
  support zip SHAは`a4095a5e1ece9c56b6b59ab57cb921c1a1bf9d1c4f027e2978e276ff583611f7`、
  埋め込みconfig SHAは`647d8796b5ba5c0462b66361f5e0879ca6a3c9afee2279c3be31ae493ba9f20b`。
  埋め込みconfigでv3 SHA、`run_approved=false`、retry approval必須を確認し、旧v2 SHAがないことを確認した。

## 2026-07-19 Kaggle train version 3再実行承認

ユーザーの「実行してください」を、次の限定scopeへの明示承認として記録した。

- active variant: 1 (`fold_safe_formation_74_addonly`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計: 15 GPU boosters
- saved corrected exp264 control再学習: 0
- corrected parent: Stage D v3 RMSE `8.460811` / OOF SHA `b11c5005...9ae2`
- approved scope: `fold_safe_formation_74_addonly_1_variant_3_lgbm_configs_5_folds_15_gpu_boosters_no_control_retraining_train_version_3_only`
- inference / submission: disabled / disabled

version 1 / 2はいずれもbooster fit前に停止しており、今回が15-booster学習の再実行承認である。
推論・提出は承認範囲外。ユーザーの前回指示どおり継続status pollingは行わず、push後の起動確認だけ行う。

### version 3 push前package監査

- strict exp / template validation、専用tests 6件、py_compile、ruff、Jupytext test: PASS。
- 再計算した実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0。
- exp264正規metricsとのcorrected Stage D v3 OOF SHA / RMSE照合: PASS。
- metadata: 同じcanonical id/title、private / T4 / internet off / `run_on_push=true` / kernel sources 4件。
- support zip SHA256: `ea3d6caaddf95d30f99f9128702f46d8fc3b5b82ebf2d75af6be40a741d817d0`。
- 埋め込みconfig SHA256: `378a805895bb1835349d80a16e0cad7199fa7d61bc1040714328a17a0a095fa4`。
- 埋め込みconfigで`run_approved=true`、approved version 3、corrected v3 SHA、15 boosters、
  control再学習なし、inference / submission無効を確認し、旧v2 SHAが存在しないことを確認した。

### Kaggle train version 3 push

- 同じcanonical kernel `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`へ
  `--accelerator NvidiaTeslaT4`を明示してpushし、`Kernel version 3 successfully pushed`を確認した。
- URL: `https://www.kaggle.com/code/kentookumura/exp287-foldsafe-form74-addonly-exp264-train`
- push後pull: PASS、Kaggle kernel `id_no=127856426`。
- Kaggle側metadata: private / `NvidiaTeslaT4` / internet off / competition source 1件 / kernel sources 4件。
- 初回statusは`KernelWorkerStatus.RUNNING`。
- ユーザーの前回指示に従い継続pollingは開始していない。kernel version 3自体は継続実行中。

## 2026-07-19 Kaggle train version 3失敗診断

- ユーザーの失敗連絡後に同一canonical kernelのstatus / logsを取得し、version 3が
  `KernelWorkerStatus.ERROR`と確定した。
- traceback到達は516.087秒、kernel終了は約523秒。exp218 clean surface生成と修正版Stage D v3
  OOF SHA gateは通過したが、saved control truth整合確認で停止し、LightGBM fitは0 / 15 boostersだった。
- clean 273 allowlistには`last_known_tvt`がmodel featureとして含まれる一方、projection側でも
  context列として先頭に指定していた。このため同じ列ラベルを2回選択し、
  `last_known_tvt`が`(3783989, 2)`、`target`が`(3783989,)`となってbroadcast errorになった。
- requested column順をfirst-occurrenceで一意化する`select_unique_columns()`を追加し、元DataFrame自体に
  duplicate labelがある場合はfail-closedで拒否する。preflightとtrainのclean projectionを同じhelperへ統一し、
  saved-control検証にもduplicate-source guardを追加した。
- `last_known_tvt`がcontextとmodel featureの両方に現れても1列だけ選択されること、および曖昧な
  source schemaを拒否することの回帰testを追加した。専用testsは8件PASS。
- version 3承認scopeは失敗runで消化済みとして`execution.run_approved=false`、
  `retry_approval_required=true`へ戻した。version 4は未承認・未pushである。
- version 4待機packageを同じcanonical id/title、private、T4、internet off、
  `run_on_push=false`でprepareした。support zip SHA256は
  `cff2351b9af370d2f05335c477d040950b938aa587470aebdddf02ecb48c2092`、
  埋め込みconfig SHA256は`c203562efdd38c28eb90c02f43be71dbc2d25a708c6308a4e8118504193fa9ba`、
  helper SHA256は`15f3b8b2fa48793a5c2cf19fc9d77e10e9fe6dfb999196156ca3e484814101f1`。
  package内configの`run_approved=false`、retry approval必須、last failed version 3と、
  package内helperのprojection修正を確認した。

## 2026-07-19 Kaggle train version 4再実行承認

ユーザーの「実行してください」を、次の限定scopeへの明示承認として記録した。

- active variant: 1 (`fold_safe_formation_74_addonly`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計: 15 GPU boosters
- saved corrected exp264 control再学習: 0
- corrected parent: Stage D v3 RMSE `8.460811` / OOF SHA `b11c5005...9ae2`
- approved scope: `fold_safe_formation_74_addonly_1_variant_3_lgbm_configs_5_folds_15_gpu_boosters_no_control_retraining_train_version_4_only`
- inference / submission: disabled / disabled

version 3はbooster fit前に停止している。version 4はduplicate projection修正後の同一scientific contractを
再実行するもので、推論・提出は承認範囲外。継続status pollingは行わず、push後の起動確認だけ行う。

### version 4 push前package監査

- 専用tests 8件、ruff、py_compile、Jupytext test、strict exp validation、template validation: PASS。
- 再計算した実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0。
- metadata: 同じcanonical id/title、private / T4 / internet off / `run_on_push=true`、
  competition source 1件、kernel sources 4件。
- support zip SHA256: `c7d4338380076cb151cf2af4261b10799e599bd9df323707ca6b377e35c503fb`。
- 埋め込みconfig SHA256: `64d420a853f0807e361dece3dd95163fb145370539c5e74810bf0138be60b77f`。
- helper SHA256: `15f3b8b2fa48793a5c2cf19fc9d77e10e9fe6dfb999196156ca3e484814101f1`。
- 埋め込みconfigで`run_approved=true`、approved version 4、15 boosters、control再学習なし、
  inference / submission無効を確認した。package内helperがprojection修正版と一致することも確認した。

### Kaggle train version 4 push

- push前に同じcanonical kernelをpullし、`id_no=127856426`、private、T4、internet offを確認した。
- 同じcanonical kernel `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`へ
  `--accelerator NvidiaTeslaT4`を明示してpushし、`Kernel version 4 successfully pushed`を確認した。
- URL: `https://www.kaggle.com/code/kentookumura/exp287-foldsafe-form74-addonly-exp264-train`
- push後pull: PASS。Kaggle側metadataは同じ`id_no=127856426`、private、`NvidiaTeslaT4`、
  internet off、competition source 1件、kernel sources 4件。
- 初回statusは`KernelWorkerStatus.RUNNING`。
- ユーザーの既存指示に従い、継続status pollingは開始していない。version 4自体は実行中。

## 2026-07-19 Kaggle train version 4失敗診断

- ユーザーの失敗連絡後に同一canonical kernelのstatus / logsを取得し、version 4が
  `KernelWorkerStatus.ERROR`と確定した。
- traceback到達は533.925秒。修正版parent OOF整合とhidden-like SHA確認までは通過したが、
  fold cache生成前の`FormationReferenceCatalog.from_raw()`で停止し、LightGBM fitは0 / 15 boostersだった。
- 最初の失敗wellは`03a935ae`で、ANCCが全行欠損していた。全773 train wellsを監査すると、
  6 formation完全行がないplane利用不可wellは8、dense ANCC利用不可wellは7だった。
- 元exp072 replayは利用不可wellをimputerごとに`continue`しており、target well自体はformation列を
  読まずに利用可能referenceだけから生成する。exp287の「全reference wellがplaneとdenseの両方で
  利用可能」という追加仮定が誤りだった。
- catalog構築をplane / denseで独立にskipする元replay準拠へ修正し、各outer foldについて
  requested / available / missing reference well数・集合SHA・missing一覧を証拠へ追加した。
  利用可能plane wellsが`k+1`未満、dense rowsが`k+1`未満の場合は引き続きfail-closedとする。
- 専用tests 9件、ruff、py_compile: PASS。実データ全773 wellsの最小catalog確認で
  plane 765 wells、dense 766 wells / 45,960 sampled rowsを構築し、欠損一覧が事前監査と一致した。

## 2026-07-19 Kaggle train version 5修正・push承認

ユーザーの「コード修正してpushまで行ってください」を、修正実装と次の限定scopeへの
明示承認として記録した。

- active variant: 1 (`fold_safe_formation_74_addonly`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- 合計: 15 GPU boosters
- saved corrected exp264 control再学習: 0
- approved scope: `fold_safe_formation_74_addonly_1_variant_3_lgbm_configs_5_folds_15_gpu_boosters_no_control_retraining_train_version_5_only`
- inference / submission: disabled / disabled

version 4はbooster fit前に停止しており、version 5もscientific contractとGPU量は変更しない。
推論・提出は承認範囲外。push後は起動だけ確認し、継続status pollingは行わない。

### version 5 push前package監査

- 専用tests 9件、全repo tests 231件、ruff、py_compile、train / inference Jupytext test、
  strict exp validation、template validation: PASS。
- 実データ773 wellsのcatalog最小再現: requested 773、plane available 765、dense available 766、
  dense sampled rows 45,960。利用不可一覧と事前欠損監査は一致した。
- 再計算した実行量: 1 variant × 3 configs × 5 folds = 15 GPU boosters、control再学習0。
- metadata: 同じcanonical id/title、private / T4 / internet off / `run_on_push=true`、
  competition source 1件、kernel sources 4件。
- support zip SHA256: `9686737f5ba9ae46f219f9349b2cb8819f1c665cab6ea9d8ddc85eaf69b6f1eb`。
- 埋め込みconfig SHA256: `b06427c7b2484ddb85455f3b38a7c92f38e441c575080cb2bfab114ffb4fc8a0`。
- helper SHA256: `60a2869483647207ee1c6de708873ef4ee50fd267dd0f4363408036e474cea76`。
- package内configでapproved version 5、15 boosters、control再学習なし、inference / submission無効、
  package内helperでplane / dense別availability policyとreference集合証拠を確認した。

### Kaggle train version 5 push

- push前に同じcanonical kernelをpullし、`id_no=127856426`、private、T4、internet offを確認した。
- 同じcanonical kernel `kentookumura/exp287-foldsafe-form74-addonly-exp264-train`へ
  `--accelerator NvidiaTeslaT4`を明示してpushし、`Kernel version 5 successfully pushed`を確認した。
- URL: `https://www.kaggle.com/code/kentookumura/exp287-foldsafe-form74-addonly-exp264-train`
- push後pull: PASS。Kaggle側metadataは同じ`id_no=127856426`、private、`NvidiaTeslaT4`、
  internet off、competition source 1件、kernel sources 4件。
- 初回statusは`KernelWorkerStatus.RUNNING`。
- 継続status pollingは開始しておらず、この時点ではversion 5が実行中だった。

## 2026-07-20 Kaggle train version 5完了監査

ユーザーの完了連絡後に1回だけstatusを確認し、canonical kernel version 5が
`KernelWorkerStatus.COMPLETE`であることを確認した。継続監視は再開していない。

### 実行量とruntime

- active variant: 1 (`fold_safe_formation_74_addonly`)
- LightGBM configs: 3 (`0`, `1`, `2`)
- folds: 5
- completed / planned GPU boosters: `15 / 15`
- saved corrected exp264 control再学習: 0
- Kaggle log runtime: `25282.477 sec`（約7時間1分22秒）
- rows / wells / final features: `3,783,989 / 773 / 421`

### CVとscope

- saved parent exp264 RMSE: `8.460811237612477`
- exp287 fold-safe formation RMSE: `8.136708220359452`
- delta new - parent: `-0.3241030172530248 ft`
- improved folds: `5 / 5`
- fold delta: `-0.397725 / -0.053862 / -0.355761 / -0.344006 / -0.465340 ft`
- near 0-250 / mid 250-1000 / 1000+ delta:
  `-0.035501 / -0.051869 / -0.365635 ft`
- hidden-like spatial / typewell-purged delta:
  `-0.620547 / -0.605988 ft`

### Promotion guard

pooled、fold、距離bucket、hidden-likeの各checkはPASSしたが、well-level safetyの2 checkがFAILし、
総合判定は`train_complete_guard_failed`となった。

- worst well: `fb03ae90`
- parent / new RMSE: `29.631678 / 37.860088`
- worst-well delta: `+8.228409822385604 ft`（固定上限`+0.25 ft`）
- +1 ft悪化well数: `135 -> 140`
- +3 ft悪化well数: `39 -> 40`
- +5 ft悪化well数: `14 -> 19`

事前固定guardを結果に合わせて緩和しない。同一OOFでのfeature/grid/threshold救済も行わず、
train完了時点ではcurrent-test生成、inference、submissionを無効のままbranchを閉じた。
後段の明示overrideによるinferenceとユーザー提出は、このtrain-side判定を変更しない。

### 取得証拠

Kaggle output全体はformation cacheとmodelsを含み巨大なため取得しなかった。CLIの
`--file-pattern`でmetrics、fold/bucket/hidden/by-well、feature importance、relationship audit、
formation/model/reproducibility manifest、raw schema auditだけを選択取得して監査した。

- OOF SHA256: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- model manifest SHA256: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- metrics SHA256: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- fold metrics SHA256: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- bucket / hidden / by-well SHA256:
  `60013cac...1069 / 5dde2227...9008 / 3562cec1...024d`
- formation fold manifest SHA256: `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- formation relationship audit SHA256: `868cc2bc3d8ea57103c70a2c150f240a29cc4d0087595d9fc4d68e864f0c86a3`
- raw schema audit SHA256: `45d0bf77b1893adfce74921f4427c4ca5ba6d95c69326cbbd35abb766e502a41`
- train run submission generated: false

formation gain上位は`tvt_dense50_d`、`tvt_densew_d`、`tvt_dense_d`、`dense_nb_std`、
`form_mean_d`。relationship auditは740行、exact duplicate 0、pruned 0で、report-only契約を維持した。
選択取得した10ファイルはローカルでbyte SHA256を再計算し、reproducibility manifest記載値と全件一致した。

## 2026-07-20 inference-only override / push前記録

ユーザーの`inferに進んでください`を、固定promotion guard FAILを保持したまま保存済みmodel inferenceだけを
実行する明示overrideとして記録した。train-side promotion、guard緩和、再学習、competition submitへは
拡張しない。`submission.csv`はKaggle outputの形式検証用にだけ生成する。

### 実行量と入力契約

- active inference variant: 1 (`fold_safe_formation_74_addonly`)
- saved Stage C selector models: 40（5 outer × 2 objectives × 4 inner）
- saved exp287 TVT models: 15（3 configs × 5 folds）
- inferenceでfitするselector / TVT booster: `0 / 0`
- final feature surface: clean 273 + outer-matched compact 74 + formation 74 = 421
- train model manifest SHA256: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- train model logical feature schema SHA256: `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`
- current test formation: all 773 train wellsをreferenceにfitし、raw test targetのformation列は読まない。
  trainと同名target wellはそのtrain referenceをqueryからself-excludeする。

### Fail-closed範囲

- train guardは`false`のまま保持し、worst well `+8.228410 ft`と+1/+3/+5 ft悪化well数増加を記録する。
- `submit_to_kaggle=false`、`competition_submit_authorized=false`、`submission_enabled=false`を
  configとnotebookで検証する。
- raw-test候補、Stage C bundle、selector schema/catalog、clean allowlist、exp287 model manifest/model、
  421列順、current-test formation finite/alignmentをSHAまたはlogical contractで拒否可能にする。

### 実装・静的検証

- 親exp264 inference v4のhidden-safe raw-test再生成を同一expへ移植し、exp287 all-train-reference
  formation surfaceと15 saved model inferenceを追加した。
- canonical Jupytext source / notebookへ採用した。ローカルnotebook実行は行っていない。
- `tests/test_fold_safe_formation_pipeline.py`: `10 passed`。
- `py_compile`、`ruff --select F821,F401,E9`、candidate / canonical Jupytext test: PASS。
- competition submitは今回の承認範囲外。Kaggle inference push後も継続監視は開始しない。

### push前package監査

- kernel id / title: `kentookumura/exp287-foldsafe-form74-addonly-exp264-infer` /
  `exp287 foldsafe form74 addonly exp264 infer`。
- metadata: private、CPU、internet off、`run_on_push=true`、competition source 1、
  kernel sources 8、Stage C selector dataset source 1。
- package notebook SHA256: `e44bab6ed01f46b5d906e008b9522e5a0b05e1990db725b93e4025559dcb57ba`
- package config SHA256: `96af61ba8264316aef99ad2a1b49d4d2e37aa0e99ec3f745e137f2b797dff9de`
- package metadata SHA256: `33105b7e3579a8eda74123ed71147ec44a4e06ff9f7b8c31ebf1abb8d645b10b`
- embedded support ZIP SHA256: `d29436fc036b1f1984eba142f0088e6dbfbebaf1c2d4c0e263626ba4bd884f17`
- embedded support files: 38。20 bootstrap dependency destinations欠損0、manifest byte/SHA mismatch 0。
- embedded formation helper SHA256: `32785414a359b5190afb364bd3f956aae37ba93f58a8333546f2f91b5163be57`
- Kaggle kernel listで同一slugは`Not found`。新規kernelとしてpushする。

### Kaggle inference version 1 push

- `kentookumura/exp287-foldsafe-form74-addonly-exp264-infer`へpushし、
  `Kernel version 1 successfully pushed`を確認した。
- URL: `https://www.kaggle.com/code/kentookumura/exp287-foldsafe-form74-addonly-exp264-infer`
- push時刻: `2026-07-20 09:43:01 JST`、Kaggle `id_no=127952811`。
- push後pull metadata audit: PASS。private、CPU (`machine_shape=None`)、internet off、competition source 1、
  kernel sources 8、dataset source 1がpackage契約と一致した。
- remote notebook内support ZIP SHA256はpackageと同じ
  `d29436fc036b1f1984eba142f0088e6dbfbebaf1c2d4c0e263626ba4bd884f17`、38 files。
- remote embedded configもguard false、40 selector、15 TVT model、booster 0、submit 3 flags falseでPASS。
- 初回status: `KernelWorkerStatus.RUNNING`。
- ユーザー方針どおり継続監視は開始しない。完了連絡後にoutputを取得し、`kaggle-submit-check`で
  `submission.csv`を検証するが、competition submitは別の明示指示がない限り行わない。
- push後の全repo testは`342 passed / 1 skipped / 2 failed`。2 failureはいずれも今回未変更のexp296
  config状態とtest期待値の既存不一致で、exp287専用10 testsとstrict validationはPASSしている。

## 2026-07-20 inference version 1完了 / scoring記録

ユーザーのscoring完了連絡後に、canonical inference kernelを1回確認した。

### Kaggle実行結果

- kernel / version / id_no:
  `kentookumura/exp287-foldsafe-form74-addonly-exp264-infer` / 1 / `127952811`
- status: `KernelWorkerStatus.COMPLETE`
- notebook内部runtime: `448.386 sec`
- Kaggle log終端: `476.945 sec`
- rows / wells: `14,151 / 3`
- saved selector / TVT models: `40 / 15`
- inference-time booster training: `0`
- final surface: clean 273 + compact 74 + formation 74 = 421

### Output / submit-check

Kaggle outputを`/tmp/exp287-infer-v1-output.2ThBRv`へ取得し、常設のsubmission copyは実験配下へ
作らなかった。skill checkerとrepo checkerはともにPASS。

- row / columns: `14,151 / id,tvt`
- sample header / row count / ID order: PASS
- duplicate ID / empty / NaN / Inf: 0
- prediction / submission row alignment: PASS
- feature schema: 421 unique（clean 273 / compact 74 / formation 74）
- formation surface: `(14,151, 76)`、74 feature全finite、target formation read 0
- submission SHA256: `deb46704998c2365cbdb91c20acd7ffdfefe0614cb5f2deb633eb8efd0ff8ca6`
- prediction decompressed SHA256: `eea88958df27dafe595a8f14bea4df980204143b6d1f7c01e65b98069c0daebc`
- formation Parquet SHA256: `d5363041a9a8d48fcca29e6529f3a636e3e2cd0ba2a7d98bbcccc3d53365ab80`
- formation logical SHA256: `cc974f8cc4bd3976b42767fc690a8085389d39d249d73ff3f8e6bdf0c44c9d8c`
- inference feature schema SHA256: `aa7a36c8341496f893f34c12210d307c599fa556f10b22b39cd4842b79a71293`

### Public LB

Kaggle提出履歴の最新`ref=54842141`（submitted `2026-07-20 01:09:32.693000 UTC`）は
`SubmissionStatus.COMPLETE`、Public LB `7.530`。ユーザー完了連絡と時系列からexp287 version 1へ
紐づけ、`submissions/SUBMISSIONS.md`の`v071`へ記録した。agent/notebookはsubmit APIを呼んでいない。

exp264 ML anchor `7.562`から`-0.032`改善したため、ML routeのPublic-LB anchorをexp287へ更新する。
別routeのexp082 ensemble 7.601も-0.071で上回るが、ensemble anchorはexp082に維持する。
train-side CV guardはworst well `+8.228410 ft`と悪化well数増加によりFAILのまま保持し、
LB anchor更新をtrain-side promotionとは扱わない。

### Route分類修正

ユーザー指示によりrouteを`ensemble`から`ml_model`へ修正した。親exp264由来のPF/HMM/Beam候補は
補助compact meta featureであり、direct blendやhard-pathを行わず、formation add-only後の最終予測は
downstream LightGBMが生成する。CV、LB、生成物、SHA、guard判定は変更しない。
`kaggle/output/`配下の実行済み生成物は、記録済みSHAを保全するため書き換えない。
