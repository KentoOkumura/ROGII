# exp413_scale5_likpf_full_replacement_on_exp335 セッションノート

## 目的

現行Public-LB reference anchorのexp335で使う`likpf_mean` primitiveを、
exp404の`likpf_scale_5_x1p0`へ候補・特徴量・selector・downstream全段で
一貫して置換するtrain-side実装候補を作成する。

## 現在の状態

- Route: ml_model
- 状態: Stage 0 / Stage C / Stage S / Stage D PASS・推論実装承認待ち
- 親/control: exp335、saved OOF RMSE `8.146107755881022` / Public LB `7.517`
- replacement evidence: exp404 scale5 x1.0 RMSE `10.914522073423171`、
  arithmetic mean `11.59489788373621`、差`-0.6803758103130392 ft`
- 実行実績: 1 variant / 40 CPU selector + 20 CPU signed selector +
  15 GPU downstream = 75 boosters
- control再学習: 0
- train新規PF: 0
- final TVT CV: `7.884802794404715`
- LB: まだなし
- 正規Notebook採用: 未承認。既存placeholderは未変更
- Kaggle package/push/run: Stage 0 / Stage C version 3、Stage S version 1、
  Stage D version 2 COMPLETE / PASS
- Stage C 40 CPU: version 3 COMPLETE / score・leakage PASS
- Stage S 20 CPU: version 1 COMPLETE / technical・score PASS
- Stage D 15 GPU: version 2 COMPLETE / technical・primary gate PASS
- 推論実装・実行、submission生成、外部提出: 未承認

## コマンドログ

### 2026-07-26 設計

- `make new-steering EXP=exp413_scale5_likpf_full_replacement_on_exp335`
- `make new-exp EXP=exp413_scale5_likpf_full_replacement_on_exp335`
- `docs/06_reproducibility.md`、親exp335/exp264、根拠exp404、
  `KAGGLE_DIRECTION.md`、`experiment_summary.md`を確認した。
- ユーザー承認はbacklog、実験scaffold、steering、設計確定まで。
  実装、正規Notebook編集、Kaggle package/push/run、推論、提出は行っていない。

### 2026-07-26 train-side実装

- ユーザーの`exp413を実装してください`をtrain-side実装承認として記録した。
- `src/likpf_full_replacement.py`を追加し、frozen exp404 gzipのraw /
  decompressed / logical / schema SHA、strict row join、replacement overlay
  cache、5 changed / 7 unchanged candidate、formula parityを実装した。
- clean273は旧22 named columnの個別patchをせず、exp218 full sourceへ
  replacement primitiveを入れ、U projection、exp145 multi-observation /
  saved exp111 learned-likelihood、GRWRを再生成して固定allowlistを適用する。
- corrected exp264 Stage Cへcache factoryを追加し40 CPU selector、
  exp335 Stage Sへ同じoverlayと20 CPU signed selectorを接続した。
  selectorのscore qualityは設計どおりreport-only、technical/leakageは必須。
- Stage Dはsaved exp335 controlをSHA固定で読み、rebuilt
  clean273 + compact74 + signed23 = final370を3 configs × 5 foldsの15 GPU
  boostersで学習し、固定primary gateとby-well tail readoutを保存する。
- Stage 0/C/S間はpreflight、semantic manifest、frozen input、
  partition/model/compact manifestのSHA lineageを相互検証し、誤った古いartifact
  rootを選んだ場合は次段へ進まない。
- `exp413_..._compact_selfcontained_train.py`から同名`.ipynb`候補を生成した。
  722行・9節で、親exp335候補398行・7節、exp264正規source465行・7節を参照し、
  replacement固有のStage 0/C/S/Dと依存再構築をNotebook上で追跡可能にした。
- 正規`exp413_..._train.ipynb`とinference placeholderは上書きしていない。

### 検証

- `python -m py_compile`: PASS
- `ruff --select F821,F401,F811`: PASS
- 専用test: `7 passed`
- 親exp264/exp335回帰test: `25 passed`
- Jupytext `--to ipynb` / `--test`: PASS
- `task validate-exp ...`: 環境に`task`がなく実行不可
- 同等コマンド
  `uv run python scripts/validate_experiment.py --experiment exp413_scale5_likpf_full_replacement_on_exp335`:
  strict PASS
- Kaggle Notebook、学習、PF、推論、提出は実行していない。

### 2026-07-27 Stage 0実行承認

- ユーザーの`実行してください`を、直前に別承認対象として示した
  0-booster `replacement_preflight`のpackage/push/run承認として記録した。
- 今回の実行量はactive variant 1、model/config 0、fold partition 5、
  CPU booster 0、GPU booster 0、control再学習0、train PF well-run 0。
- Stage C 40 CPU、Stage S 20 CPU、Stage D 15 GPUはrun flagをfalseのまま維持し、
  今回は起動しない。
- 正規train placeholderは上書きせず、検証済みcompact self-contained候補から
  Stage 0専用notebook packageを生成する。
- `scripts/prepare_kaggle_notebooks.py`へ`replacement_preflight` kindを追加し、
  `exp413_..._replacement_preflight.ipynb`を生成した。正規train notebookは未変更。
- 予定kernel: `kentookumura/exp413-scale5-likpf-replacement-preflight`
  （title: `exp413 scale5 likpf replacement preflight`）。CPU、internet off、
  run-on-push、入力は公式competition、exp263 cache、exp404 frozen Datasetだけ。

### 2026-07-27 Stage 0 Kaggle version 1

- kernel id: `kentookumura/exp413-scale5-likpf-replacement-preflight`
- kernel id_no: `128773100`
- version: 1
- metadata: CPU、internet off、exp263 kernel + exp404 Dataset + competition input。
- 結果: ERROR。bootstrapと入力解決は通過したが、replacement cache書き込み前の
  `last_known_tvt`整合で停止した。
- 原因監査: SHA固定済みローカル入力で全5 foldを比較し、ID miss 0、
  `md_since` max差0。親cacheはfloat32、exp404 frozen列はfloat64で、
  raw float64比較のmax差は全fold `0.00046875000043655746 ft`だった。
  float32へ量子化した不一致は0件で、意味値差ではなく保存dtype差だった。
- 修正: 許容幅を緩めず、親cacheの保存精度float32へ両入力を量子化した後に
  exact equalityを要求するよう整合guardを変更した。0.01 ft差を拒否する
  regression testも追加した。

### 2026-07-27 Stage 0 Kaggle version 2

- version: 2
- 結果: ERROR。version 1停止点は通過し、全5 foldのreplacement cacheと
  selector probeを処理した後、186秒でformula parity guardに停止した。
- 原因: candidate bank本体は親contractどおり、各weight積と加算ごとに
  float32へ量子化する。一方、監査側だけfloat64で理想実数式を再計算し、
  float32保存値へ`1e-6 ft`を要求していたため、算術dtype契約が不一致だった。
- 修正: contractの重み・積・加算をcandidate bank本体と同じfloat32順序で
  再計算し、保存済み候補値とのexact parity（max差0）を要求する。

### 2026-07-27 Stage 0 Kaggle version 3

- kernel: `kentookumura/exp413-scale5-likpf-replacement-preflight`
- kernel id_no / version: `128773100` / `3`
- status: COMPLETE / technical PASS
- runtime: notebook出力完了`199.362 sec`、Kaggle nbconvert完了`210.182 sec`
- 実行量: active variant 1、fold partition 5、model/config 0、
  CPU/GPU booster 0、control再学習0、PF/HMM/Beam well-run 0。
- 入力: exp263 cache 3,783,989 rows / 773 wells / 5 folds、
  exp404 frozen prediction raw/decompressed/logical/schema SHAは全一致。
- 候補: 12固定、5 changed / 7 unchanged、changed rows 3,698,554。
  unchanged / formula / parent old-mean parity max absはすべて`0.0 ft`。
- selector probe: 各fold 1,024 base rows / 12,288 candidate-long rows /
  88 features。compact schemaは74 features。
- Stage D、inference、submissionは実行されていないことをログで確認した。
- outputを`kaggle/output/stage_0`へ取得し、
  `verify_replacement_stage_0_root`を再実行してPASSした。取得量は約80 MB。
- executed config SHA:
  `c06d0468bdb2798628989d362ad6729ab3bacafd78f3b883e5bccf088ff6f43a`
- executed source SHA:
  `470f04d97a840bda3535a46888a39c7d0116f8acd2fdbab64643374ac3aaded8`
- Kaggle log SHA:
  `377c88772826c15eff4784004127c4d3b8528d974cfe20947da2339412acfbb3`
- preflight file SHA:
  `c1c536daaa4d0250578ba427882745f131223ed988f7b3adc812ed7daa33b258`
- semantic manifest file / logical SHA:
  `b3c98af2198b92756ac1db342e34bcf7bfc31ee54d9b2d85e5a01c0141fd34c7` /
  `8b2ff389467abb2480eeb00d55e8898fa777dbda5c0523b35ac7709bff425fdf`
- 完了後、全run flagをfalseへ戻して意図しない再実行をfail-closedにした。
- local Kaggle packageも`run_on_push: false`かつ全run flag falseで再生成し、
  完了済みversion 3の意図しない再pushを防いだ。実行時config/sourceは取得済み
  Stage 0 outputと上記SHAを正とする。

### 2026-07-27 Stage C実行承認

- ユーザーの`Stage Cに進んでください`を、Stage C専用package/push/runの
  明示承認として記録した。
- 実行scopeは1 replacement variant、2 objectives
  (`pred_abs_error`, `p_within10`) × outer 5 × inner 4 =
  40 CPU selector boosters。
- LightGBM configは親corrected exp264 Stage C v6から変更せず、seed 42、
  deterministic / force-col-wise / threads 8、Kaggle CPU、internet off。
- 期待出力は40 models、25 compact partitions、18,919,945 compact rows、
  outer-valid score 45,407,868 candidate-long rows、selector88→compact74。
- saved parent/control再学習0、GPU booster 0、PF/HMM/Beam well-run 0。
- Stage S 20 CPU、Stage D 15 GPU、inference、submissionのauthorizationと
  run flagはfalseのまま維持し、今回起動しない。
- 入力Stage 0はkernel
  `kentookumura/exp413-scale5-likpf-replacement-preflight` version 3、
  semantic manifest SHA
  `b3c98af2198b92756ac1db342e34bcf7bfc31ee54d9b2d85e5a01c0141fd34c7`
  を再検証してPASSした。
- Stage C kernel id/titleは
  `kentookumura/exp413-scale5-likpf-selector-train` /
  `exp413 scale5 likpf selector train`。初回pullは403、mine searchは
  `Not found`で、既存同名kernelなしを確認した。
- package metadataとbootstrapはCPU、internet off、run-on-push、
  上記2 kernel sources + exp404 Dataset + competition sourceで一致した。
- push予定config SHA:
  `5674744b058b2b758e59164839a2ca1e178104da47774b35f3d0db689fd81df2`
- push予定source SHA:
  `470f04d97a840bda3535a46888a39c7d0116f8acd2fdbab64643374ac3aaded8`

### 2026-07-27 Stage C Kaggle version 1開始

- kernel: `kentookumura/exp413-scale5-likpf-selector-train`
- kernel id_no / version: `128776527` / `1`
- post-push metadata: private、CPU、internet off、exp263 + exp413 Stage 0
  kernel、exp404 Dataset、competition source。
- push直後status: `KernelWorkerStatus.RUNNING`
- 実行config/source SHAは上記push予定値と一致する。
- 重複実行防止のため、push成功直後にlocal
  `execution.run_flags.nested_selector_train`をfalseへ戻した。

### 2026-07-27 Stage C Kaggle version 1失敗

- status: `KernelWorkerStatus.ERROR`
- 40/40 CPU boosterの学習ログとouter 5の学習完了までは確認した。
- 最終`nested_selector_metrics.json`保存後、
  `/kaggle/working/artifacts/reproducibility_manifest.json`を更新しようとして
  `FileNotFoundError`で停止した。
- 原因: 親exp264のStage Cは同一runのStage Aが先にreproducibility manifestを
  作るが、exp413は別runのStage 0からschemaをコピーする設計で、そのseed
  manifestをStage C outputへ作成していなかった。
- Kaggle file listingは空で、ERROR runの40 model / 25 partition成果物は
  回収不能だった。
- 修正: `run_stage_c`前にStage 0 preflight / semantic manifest /
  frozen prediction、selector schema/catalog、compact schema、親config /
  candidate contract、seed、cost contractのSHA evidenceを持つ
  `reproducibility_manifest.json`を先に生成する。
- 科学条件は変更しない。1 variant、2 objectives、outer 5、inner 4、
  40 CPU boosters、control再学習0、PF/HMM/Beam 0のままversion 2を実行する。

### 2026-07-27 Stage C Kaggle version 2開始

- kernel: `kentookumura/exp413-scale5-likpf-selector-train`
- kernel id_no / version: `128776527` / `2`
- 実行config SHA:
  `50dbd3c46fb8014738d37834178499344e6a5c62413df7e3175a83856c7e3721`
- 実行source SHA:
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- package監査: Stage Cだけrun flag true、Stage S/D false、1 variant、
  2 objectives、outer 5、inner 4、40 CPU boosters、control再学習0、
  PF/HMM/Beam 0、CPU、internet off。
- 修正後検証: Jupytext round-trip、py_compile、Ruff、
  strict experiment validation PASS、専用/親回帰test `33 passed`。
- push成功後、重複実行防止のためlocal/packageを全run flag false、
  `run_on_push: false`へ戻す。

### 2026-07-27 Stage C Kaggle version 2失敗

- status: `KernelWorkerStatus.ERROR`
- bootstrap 41 filesの展開後、学習前authorization guardで即停止。
  booster実行数は0。
- 原因: package directory直下の`config.yaml`だけをpush直前にtrueへしたが、
  Kaggleへ送られるNotebook内bootstrap zipはpackage準備時のfalse configを
  保持していた。
- 修正: local configのStage C run flagを一時trueへしてからpackage /
  Notebookを再生成し、Notebook先頭cellを一時directoryへ展開して、埋込configの
  Stage Cだけtrue、Stage S/D falseをpush前に監査する。push後にlocal/packageを
  再封印する。
- version 1で修正したreproducibility manifest seedは維持する。variant、
  objectives、fold、LightGBM config、40 CPU boosterの科学条件は変更しない。

### 2026-07-27 Stage C Kaggle version 3開始

- kernel: `kentookumura/exp413-scale5-likpf-selector-train`
- kernel id_no / version: `128776527` / `3`
- 実行config SHA:
  `a84c088da829584ab097bb5dc8ecd883362e2c8eb909f6ff1c3733c594369e1e`
- 実行source SHA:
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- Notebook先頭bootstrap cellを一時directoryへ実展開し、41 support files、
  Stage C run flagだけtrue、Stage S/D false、埋込source SHA一致を確認した。
- metadata: private、CPU、internet off、run-on-push、exp263 + Stage 0
  kernel、exp404 Dataset、competition source。
- push成功後はlocal/packageを全run flag false、
  `run_on_push: false`へ再封印する。

### 2026-07-27 Stage C Kaggle version 3完了

- status: `KernelWorkerStatus.COMPLETE`
- runtime: `6378.321285492 sec`
- 実行量: 1 replacement variant、2 objectives × outer 5 × inner 4 =
  40/40 CPU boosters。control再学習0、PF/HMM/Beam well-run 0。
- 出力: 40 model、25 compact partitions、18,919,945 compact rows、
  45,407,868 outer-valid candidate-score rows。
- score guard: PASS。expected-error MAE `3.7206340209583226`
  （prior `5.70019990923869`）、within10 logloss `0.34957913108150057`
  （prior `0.49981391647035633`）、Brier `0.10806449114014476`
  （prior `0.16070347246562533`）。3指標とも5/5 folds改善。
- leakage audit: PASS。outer-validはinner assignmentから除外、
  inner train/valid wellはdisjoint、outer-train compactはinner OOF、
  outer-valid compactは4 inner model ensemble。
- hard readoutは設計どおり無効。final TVT CVの改善判定ではない。
- model / compact manifest SHA:
  `7badcf45bccab0e2b0535ab7e7171e4c6cb7bf81e0bb115d5e283b1486b7b875` /
  `507429faa4fbc336dbc00e8edfee5a45788b8a58dbc2e15a440d5e7780d5f07f`
- outer-valid candidate-score SHA:
  `016408a6e77a3708be5cec285976b95b9f178921a51cac79780b169111c242cd`
- Stage C lineage / reproducibility manifest SHA:
  `2e0a36a7383d45a2f2aae7bba21c8160abff4a9d47e47a4d814300e729774aa1` /
  `ecf4b9a1e41199705438cf44982f417a4055a23f33e8b29e0d44c7b03093441a`
- metrics / partition / fold manifest SHA:
  `4642e1b22662d830fcd601f5d46452657e51546f1d741b1e0ae4a8e1c836168d` /
  `b79b5193beef67c022addbaff90aee8bd96995b8392d8e40d753b29095d86132` /
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`
- Kaggle log SHA:
  `fe9c8f94d7b386305e8268e28b163b2e39368931d8147b98a59a76ed2692f35f`
- 約3GBのlarge outputは取得せず、小規模metrics/manifest 7ファイルとlogだけを
  選択取得した。model数、objective、outer/inner fold、partition行数、
  manifest内部SHAをlocalで再監査して全一致。
- Stage D、inference、submissionが実行されていないことをログで確認した。
- 完了後のlocal/packageは全run flag false、`run_on_push: false`。

### 2026-07-28 Stage S signed-residual selector学習承認

- ユーザーの「次に進んでください」を、直前に次段として示した
  `signed_selector_train`のKaggle package、push、run承認として受領した。
- 実行stage: `signed_selector_train`
- active replacement variant数: 1（`scale5_x1p0_likpf_full_replacement_on_exp335`）
- signed-residual objective / LightGBM config数: 1（`signed_residual_l2` /
  `regression_l2`）
- fold構成: outer 5 × inner 4
- 今回の合計学習量: 20 CPU boosters
- 完了済みStage C selector再学習: 0 boosters
- saved exp335 control再学習: 0 boosters
- Stage D: 0 GPU boosters（未承認）
- train PF / HMM / Beam: 0 well-runs
- inference / submission生成 / external submission: 未承認
- Kaggle CPU、internet off、seed 42、`deterministic=true`、
  `force_col_wise=true`、threads 8で実行する。
- 入力は公式competition、exp263 cache、exp413 Stage 0、exp413 Stage C、
  exp404 frozen Datasetに限定する。Stage Cの40 modelと25 compact partitionは
  親kernel outputを直接mountし、再学習しない。
- 期待出力は20 models、signed23の25 compact partitions、
  18,919,945 compact rows、45,407,868 outer-valid candidate-score rows。
- technical / score gateの判定後は必ず停止する。PASSしてもStage Dの
  15 GPU boostersは別承認なしに開始しない。FAIL時もobjective/grid救済を
  同じOOFで行わない。
- 予定kernel: `kentookumura/exp413-scale5-likpf-signed-train`
  （title: `exp413 scale5 likpf signed train`）。正規train placeholderは
  上書きせず、Stage S専用のcompact self-contained notebookを生成する。
- Stage 0 / Stage CのKaggle statusはともに`COMPLETE`。対象slugのstatusは
  access denied、`kaggle kernels list --mine --search`は`Not found`で、
  自分のアカウントに既存同名kernelがないことを確認した。
- push前検証: Jupytext round-trip、py_compile、Ruff
  `F821/F401/F811`、strict experiment validation、専用＋親回帰test
  `33 passed`。
- Notebook bootstrap ZIPをコード実行せずAST/Base64/ZIPとして解析し、
  41 support files、Stage S run flagだけtrue、Stage D false、
  1 variant、1 objective、outer 5 × inner 4、20 CPU boosters、
  control再学習0、PF/HMM/Beam 0を確認した。
- embedded config SHA:
  `532372898a2a5cff0cbd99b817f82f71f0423626d79b607193cbbcbc8cca92eb`
- embedded source SHA:
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- bootstrap済みNotebook SHA:
  `06d45840514d53fa2cf6695e84c341fff0c39425b7bb728a219d571e39ba7e43`
- kernel metadata SHA:
  `38597c6893b478313de3626748dd62362d084e25d7c9fd974091318b074dc57b`
- version 1をpushし、Kaggleからsuccess応答を確認した。実行kernel URL:
  `https://www.kaggle.com/code/kentookumura/exp413-scale5-likpf-signed-train`
- 重複実行防止のため、push成功直後にlocal
  `execution.run_flags.signed_selector_train`をfalseへ戻し、packageも
  `run_on_push: false`で再生成する。
- push直後のKaggle statusは`RUNNING`。bootstrap、Stage 0 / Stage C
  lineage検証を通過し、LightGBM学習ログ（iteration 100）を確認した。
- ユーザーの「監視は止めていいです。完了したら連絡します。」に従い、
  Kaggle kernel自体は停止せず、こちらの`kaggle kernels logs -f`だけを
  終了した。完了metrics、20/20 model、technical / score gate、artifact
  SHAは未確認なので、現時点では結果として主張しない。

### 2026-07-28 Stage S Kaggle version 1完了確認

- ユーザーの完了連絡後、Kaggle status
  `KernelWorkerStatus.COMPLETE`と完了logを取得した。
- kernel: `kentookumura/exp413-scale5-likpf-signed-train`
- kernel id_no / version: `128832243` / `1`
- runtime: notebook summary完了`2972.707 sec`、Kaggle nbconvert完了
  `2984.194 sec`（約49.7分）
- 実行量: 1 replacement variant、1 signed-residual objective × outer 5 ×
  inner 4 = 20/20 CPU boosters。Stage C / saved control再学習0、
  Stage D GPU booster 0、PF/HMM/Beam well-run 0。
- 20 outer-inner組と20 unique model SHAを確認した。best iterationは
  58–147、中央値85.5。
- pooled signed-residual RMSE `8.291963292672019`、candidate別
  outer-train mean prior `10.854995765411246`、改善
  `2.563032472739227 ft`。
- fold 0–4 RMSEは`8.357041 / 8.328928 / 7.942127 / 8.069783 /
  8.739282`、priorは`10.662662 / 10.812323 / 10.530472 /
  10.701568 / 11.539241`で、5/5 foldsが改善した。
- score gate: PASS（pooled改善、改善fold 5/5、要件4/5）。
- technical gate: PASS。candidate順、88特徴、outer-valid除外、
  inner well-disjoint、outer-train inner OOF、outer-valid 4-model
  ensemble、25 partitions / 18,919,945 compact rows /
  45,407,868 outer-valid score rowsを確認した。
- formula parity / saved exp264 top-1 value parity max abs errorはともに
  `0.0`。Stage S総合gate: PASS。
- candidate別pooledでは12候補中11候補がpriorを改善した。
  `exp226_w500_50_50`だけはRMSE `8.176647`、prior `8.053034`で
  `-0.123613 ft`の非改善。Stage S固定gateはPASSだが、Stage Dでは
  downstream RMSE / scope / tail guardで吸収できるか確認が必要。
- 小規模metrics/manifests 13ファイルとlogだけを
  `kaggle/output/stage_s/`へ取得した。20 model本体、25 compact parquet、
  45M-row score parquetは一括downloadしていない。
- executed config / source SHA:
  `532372898a2a5cff0cbd99b817f82f71f0423626d79b607193cbbcbc8cca92eb` /
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6`
- log / metrics JSON file / metrics CSV SHA:
  `1cf13bf0fd0f6a001061f9b050cf1077c5433c6eca68d822012c52cb0a07fe40` /
  `ce540b392f23e3becd0b65bceb01589424dac0b0593646c93043d75f81525410` /
  `68abf6acb658b379a3b9ad7429249d84a63f77fbdde9d412ecb813ebc9b5874d`
- model / compact / partition manifest SHA:
  `84a0047667ce9209ce63e3b9e935ff6c379d2d38b1ae4262318db94a47aaca9f` /
  `7a4282a25d7e7887e314cd3d01b8a09c81ff91ba3c1b1cf62e3197079ac93323` /
  `893d144fa1956b0cdcc4f4d75b9cfe3ebc05fd5bc54df0e7345a47f4b2c70ff1`
- signed compact schema file / logical SHA:
  `e57c6d5fde307eb26193a1c5efdb31accde94a6836df75de03100189d232021a` /
  `74abf31f057dfe29177221895e3e26c5a261e5b51defc04f081d6b140f2be44c`
- outer-valid score / Stage S lineage / reproducibility manifest SHA:
  `004b0830fdc1a7893e5fdac77a217d717c6f7673a36065f50d297fa44c426841` /
  `d0e279e2c844d557c731107a0925f2d1c27de77eb64cfad9a498ab0b10ea0c13` /
  `6acebb4461ac2df6f94c6edf1068a0ba9e42bc8c8877d83b4b3ccfa586a3e7c1`
- 取得したmanifest間、Stage 0 semantic SHA、Stage C lineage SHA、
  実行Notebookの静的bootstrap config/source SHAをlocalで照合してPASSした。
- ログ末尾で`Stage D was not executed`、`Inference executed: False`、
  `Submission generated or submitted: False`を確認した。Stage Dは
  別承認まで起動しない。
- 完了記録後にStage S packageを再生成し、status
  `stage_s_complete_waiting_downstream_gpu_train_approval`、全run flag false、
  `run_on_push: false`の封印状態を静的に再確認した。

## 変更点

- candidate count 12、ID、順序、family、legal domainを固定する。
- `likpf_mean` primitiveと、それを親に持つ4 formula/fixed slotの計5 slotを
  scale5 x1.0 sourceで再計算する。
- clean273は`likpf_mean`を名前に含む22列だけをpatchせず、全273列を
  replacement sourceから再構築する。
- selector88→compact74を40 CPU boosters、signed23を20 CPU boosters、
  final370を15 GPU boostersで再学習する。
- old arithmetic meanはparity監査専用で、candidate/model入力には残さない。
- 13候補化、add-only、x1.3、scale grid、feature subset、weight/threshold救済は禁止。

## 固定gate

- pooled improvement: saved exp335比`>=0.03 ft`
- nonworse folds: `>=3/5`
- near/mid/1000+、hidden-like spatial/typewell-purged: 各`<=+0.02 ft`
- by-well p95、worst、`+1/+3/+5 ft`悪化well数: report-only
- FAIL時: same-OOF rescueなしでclose
- PASS時: 同じexp413内のinference実装資格だけを得る。実装・実行・提出は別承認。

## 再現性メモ

- seed policy: train scale5はexp404 SHA固定生成物、current-testは
  exp072互換stable SHA256 per-well seed、modelは親と同じseed 42。
- stochastic components: saved likelihood-PF paths、CPU/GPU LightGBM。
- parallel RNG: well seedを並列実行前に固定しglobal RNGをthread内で使わない。
- CPU/GPU runtime: selector CPU deterministic/force-col-wise/threads8、
  downstream GPU DP/deterministic/force-col-wise/threads8。
- deterministic anchor: false。GPU bitwise一致は主張しない。
- input/feature schema/content SHA: 実装時にprimitive、candidate12、
  clean273、selector88、compact74、signed23、final370を段階別に記録する
  manifest処理を実装済み。Stage 0 primitive/candidate、Stage C
  selector88/compact74、Stage S signed23は生成・記録済み。clean273/final370は未生成。
- model manifest/model SHA: Stage C 40 modelとStage S 20 modelは記録済み。
  Stage Dは未生成。
- prediction/submission SHA: 未生成。将来current-testへ進む場合はcontent SHAを記録する。
- Kaggle kernel id/version: Stage 0 `128773100` / v3、Stage C
  `128776527` / v3、Stage S `128832243` / v1。

## 次のアクション

1. Stage D 15 GPUを別run承認のもと実行する。
2. Stage D primary gate PASS時だけ、同じexp413内のcurrent-test inference実装を
   別承認で開始する。

### 2026-07-28 Stage D 実行承認・push前コスト監査

- ユーザーの「Stage Dに進んでください」を、Stage D専用notebookの生成、
  Kaggle T4へのpush、完了確認、固定primary gateでの停止までの承認として
  記録する。inference実装・実行、submission生成、外部提出は承認に含めない。
- 実行variantは`scale5_x1p0_full_replacement`の1件だけ。LightGBM configは
  凍結済み`[0, 1, 2]`の3件、outer foldは5件、合計GPU boosterは
  `1 × 3 × 5 = 15`。
- saved exp335 controlは既存Stage D v2 OOF / metrics / model manifestを
  比較参照するだけで、control再学習は0 booster。Stage C 40 boosterと
  Stage S 20 boosterも再学習しない。PF/HMM/Beam well-runは0。
- downstream特徴量はclean273をreplacement sourceから再構築し、保存済み
  Stage C nested74、保存済みStage S signed23を結合したfinal370に固定する。
  13候補化、scale grid、feature subset、objective変更、same-OOF rescueは行わない。
- Stage Dで使う親の固定値を監査し、exp335と同じ
  `early_stopping_rounds=250`、`log_evaluation_period=100`、
  `matrix_copy_chunk_columns=32`をexp413 configへ明示した。
- Kaggle入力はcompetition data、exp404 frozen prediction Dataset、
  exp263、exp413 Stage 0 / C / S、exp072 replay cache、exp099 cache、
  exp111 schema/models、保存済みexp335 Stage D v2 controlに限定する。
- acceleratorは`NvidiaTeslaT4`、LightGBM modeは
  `gpu_repro_guard_dp_threads8`（GPU DP、deterministic、
  force-col-wise、threads 8）。GPUのbitwise一致は主張せず、
  input/feature/model/prediction SHAとkernel versionを記録する。
- primary gateはsaved exp335 RMSE `8.146107755881022`に対してpooled gain
  `>=0.03 ft`、nonworse folds `>=3/5`、near/mid/1000+と
  hidden-like spatial/typewell-purgedが各`<=+0.02 ft`。by-well tailは
  report-only。PASS/FAILのどちらでもこの固定gate直後に停止する。
- 予定kernelは`kentookumura/exp413-scale5-likpf-downstream-train`
  （title: `exp413 scale5 likpf downstream train`）。正規train placeholderは
  上書きせず、Stage D専用compact self-contained notebookを生成した。
- 全8件のkernel input（exp263、Stage 0 / C / S、exp072、exp099、
  exp111、exp335 Stage D v2）がKaggleで`COMPLETE`であることを確認した。
  対象slugのstatusはaccess denied、`kaggle kernels list --mine --search`は
  `Not found`で、自分のアカウントに既存同名kernelがない。
- push前検証はJupytext round-trip、py_compile、Ruff
  `F821/F401/F811`、strict experiment validation、専用＋親回帰test
  `33 passed`。
- Notebook bootstrap ZIPをコード実行せずAST/Base64/ZIPとして解析し、
  41 support files、Stage D run flagだけtrue、1 variant、
  configs `[0, 1, 2]`、5 folds、15 GPU boosters、control再学習0、
  PF/HMM/Beam 0、inference/submission全falseを確認した。
- embedded config SHA:
  `29099eeebba2c1bd7f89a6747436ae6a1ccb4a27f2ac410b03c69780112c0f56`
- embedded notebook source / `src.likpf_full_replacement` SHA:
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6` /
  `470f04d97a840bda3535a46888a39c7d0116f8acd2fdbab64643374ac3aaded8`
- bootstrap済みNotebook SHA:
  `05978d54d009e1f0dc6543597197fd935f0dc7cdcbb3d66260bd0a075eef9f73`
- kernel metadata SHA:
  `c8610346c2629dfef35a551cf0cbb9bf42509d7dd28a7a03b8a0801891219d2e`
- Kaggle version 1のpush成功応答を確認した。実行URL:
  `https://www.kaggle.com/code/kentookumura/exp413-scale5-likpf-downstream-train`
- 重複実行防止のため、push成功直後にlocal
  `execution.run_flags.downstream_gpu_train`をfalseへ戻し、packageも
  `run_on_push: false`で再生成する。

### 2026-07-28 Stage D Kaggle version 1失敗

- Kaggle statusは`ERROR`。41 support files、leakage contract、保存済み
  Stage 0 / C / Sの選択までは通過したが、clean273再構築中のexp145 source
  dynamic importで停止した。
- failure phaseは学習前`pre_training_dynamic_exp145_module_import`。
  LightGBM完了ログは0/15、saved control再学習0、回収可能なStage D
  metrics/model/OOFは0。実験のGPU booster条件はまだ実行されていない。
- 原因はPython 3.12の`dataclasses`がannotation解決時に
  `sys.modules[cls.__module__]`を参照する一方、共通`_load_module`が
  `module_from_spec`後にmodule登録せず`exec_module`していたこと。
- 修正範囲は`_load_module`のimport互換性だけ。実行前にmoduleを
  `sys.modules`へ登録し、exec失敗時は以前の状態へ戻す。dataclassを含む
  一時moduleの回帰testを追加する。variant、feature、fold、config、
  primary gate、15-booster cost contractは変更しない。
- loader修正後の専用＋親回帰testは`34 passed`。py_compile、Ruff
  `F821/F401/F811`、strict experiment validationもPASSした。v2でも
  Stage D run flagだけを一時的にtrueにしてpackageを再監査する。
- v2 bootstrap静的監査は41 support files、Stage Dだけtrue、15 GPU、
  control 0、推論/提出falseでPASSした。T4/private/internet offと入力集合は
  v1から不変。
- v2 embedded config / notebook source / loader修正済みmodule SHA:
  `67bea06a6082892cb93c6ac4e9be6dfe15665d48caca34c6b0da7878bf5a1c0e` /
  `5757eef8c0aea3ad52e5f3538a44146aec494ea0b393c192c59b423a8ac478e6` /
  `3b557b3a19c5dcc62dfe6567bfe4fdf6b47f3d2691549b87671d03763d0da566`
- v2 bootstrap済みNotebook / kernel metadata SHA:
  `988a2c55ff11dbbe2bce94319e48c0dc66386f8bf2131a776348a5533161388e` /
  `c8610346c2629dfef35a551cf0cbb9bf42509d7dd28a7a03b8a0801891219d2e`
- Kaggle version 2のpush成功応答を確認した。push直後にlocal/packageを
  全run flag false、`run_on_push: false`へ再封印する。

### 2026-07-28 Stage D Kaggle version 2完了・固定gate停止

- kernel `kentookumura/exp413-scale5-likpf-downstream-train`
  version 2（id_no `128914549`）は`COMPLETE`。最終Notebook runtimeは
  `17374.782 sec`、実行ログ最大timestampは`17386.338 sec`。
- 固定実行量どおりreplacement 1 variant × LightGBM configs 3 × outer folds 5
  = 15/15 GPU modelsを学習した。saved exp335 control、Stage C、Stage Sの
  再学習は0、PF/HMM/Beamは0。clean273 + nested74 + signed23 = final370。
- saved exp335 RMSE `8.146107755881022`に対しreplacement RMSE
  `7.884802794404715`、gain `0.26130496147630744 ft`。
  fold deltaは`[-0.080573, -0.307953, -0.132971, -0.258389, -0.510381] ft`で
  5/5 folds nonworse。
- scope deltaは0--250 `-0.020295`、250--1000 `-0.019498`、
  1000+ `-0.296706`、spatial `-0.487557`、typewell-purged
  `-0.501682 ft`。最大scope deltaは`-0.019498 <= +0.02 ft`。
  technical checks、pooled gain、fold、scopeの固定primary gateをすべてPASSした。
- report-only tailはby-well delta p95 `+1.228715 ft`、worst well
  `fa31da94 +9.033462 ft`、`+1/+3/+5 ft`悪化well数は`55/8/6`。
  LB-oriented inference候補には昇格するが、robust promotionとは扱わない。
- 15 model SHAは全件一意でconfig/fold gridを一意に網羅した。best iterationは
  min / median / max `235 / 1796 / 8230`、feature importanceは11,100行。
- final OOF / fold / scope / hidden / by-well SHA:
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d` /
  `82e70b6674f218f2892d6e5f70e327dfcbbdaf0fa5e431c4e07231009e9e2d8f` /
  `c89add97cd4cae628b79774615a717e4cfbffe7b65a4a68c58b2c2e2737948ed` /
  `eafa3546e4ea5c0d180d380f7fe2c39b5cac970ea4c8097b68b077017da1f1b8` /
  `e82c6908ed2caa9b3e5c1664bc66a3226b3bc6d9284f4863bd4fa941ae32d080`。
- feature importance / model manifest / reproducibility manifest SHA:
  `48cf4fa355fa57e235c8d396a07afbb04938ea9aca6cf5df685fc0aae0fea484` /
  `4b4f988154468ba6697cdd57c0a0c6bf7cc631e7b2bbe1f15fa8f51fdeb7c3df` /
  `5cfcf6d5fb76bd1b23782016967d95c4de86ac82489ab6f6f95a86bd41c1e472`。
- 小さい監査artifactと実行logだけを
  `kaggle/output/stage_d/`へ取得した。large model/OOF archiveは取得していない。
  local/packageは全run flag false、`run_on_push: false`に封印した。
- 固定gate直後で停止した。inference実装・実行、submission生成、外部提出は
  すべてfalseで、別承認までfail-closedを維持する。
- 完了記録を反映してStage D packageを再生成した。埋込config SHAは
  `9110b0d217fb62f3f301205e60f25f38fd6a52413cad7591504ae84f5b5a75fe`、
  sealed notebook / metadata SHAは
  `e13cbfbc12881d78aaf536999e2304428884f72248db3d24d585bf906f736776` /
  `6c2aaa32878d3e4c6f59ba2402b89724d857df0174f450e3593498ec0774a877`。
  41 support files、全run flag false、`run_on_push: false`、inference /
  submission全false、T4/private/internet off、実行量15/control 0を静的監査した。
- 最終検証は専用9 + 親回帰25 = `34 passed`、Jupytext round-trip、
  py_compile、Ruff `F821/F401/F811`、strict experiment validation、
  取得済みStage D artifact相互SHA/model grid/gate再監査をすべてPASSした。

### 2026-07-29 current-test推論承認・push前監査

- ユーザーの「推論に進んでください」を、Stage D primary gate PASS後の
  current-test推論実装、Kaggle CPU package/push/run、予測監査outputの取得と
  検証までの承認として記録する。`submission.csv`生成と外部提出は含めない。
- 正規inference placeholderは上書きせず、Jupytext percent形式の
  `*_current_test_inference.py/.ipynb`と
  `kaggle/current_test_inference/`を別に作成した。
- raw testからexp263固定の12候補、21 native-confidence列、clean273を再生成する。
  likelihood-PFは500 particles × 128 stable per-well seedsから得る同じtrajectory
  bankのtemperature-5列を`likpf_mean` semantic slotへ全面置換する。
  arithmetic meanはcontent SHA比較専用でcandidate/model入力には残さない。
- 保存済みStage C version 3の40 selector、Stage S version 1の20 signed selector、
  Stage D version 2の15 TVT modelをmanifest/model SHA照合後にCPU適用する。
  学習variant 0、LightGBM training config 0、fold学習0、合計新規booster 0。
  親control再学習、PF以外の学習、新規GPU使用も0。
- Stage D model manifest SHAは
  `4b4f988154468ba6697cdd57c0a0c6bf7cc631e7b2bbe1f15fa8f51fdeb7c3df`。
  Stage C / S model manifest SHAは
  `7badcf45bccab0e2b0535ab7e7171e4c6cb7bf81e0bb115d5e283b1486b7b875` /
  `84a0047667ce9209ce63e3b9e935ff6c379d2d38b1ae4262318db94a47aaca9f`。
- 専用10 + 親回帰9 + package helper 3 = `22 passed`。Jupytext round-trip、
  py_compile、Ruff `F821/F401/F811`、strict experiment validationもPASSした。
- bootstrap ZIPをAST/Base64/ZIPとして静的監査し、50 support files、
  current-test推論run flag true、保存model count 40/20/15、学習0、
  submission生成/submit false、CPU/private/internet off、dataset source 0、
  kernel source 11を確認した。
- embedded config / source / bootstrap済みNotebook SHA:
  `f518ef270cff279f8f2f284a0ff0b49c3e800319f9cc004f3aa6f89cb329dfa4` /
  `185129560120de0081737b2d57a12707dfd6eb171aa82f8a2ee58467b766c90b` /
  `5960af7ba1b94d94257b21b85b9d21f3c06e3a4ff3950331c195c60a257c4a2a`。
- 予定kernelは
  `kentookumura/exp413-scale5-likpf-current-test-inference`
  （title: `exp413 scale5 likpf current test inference`）。push成功後は
  local `inference.run_enabled`とpackage `run_on_push`をfalseへ戻して再封印する。

### 2026-07-29 current-test推論Kaggle version 1失敗

- kernel version 1（id_no `128975306`）は`ERROR`。runtime `406.181 sec`。
  raw test 14,151 rows、likelihood-PF 500 particles × 128 seeds、K16、
  exact/self-GR HMM、scale5 semantic replacement、candidate-long selector処理までは
  通過した。
- failure phaseはsigned-selector top1 parity toleranceのconfig lookup。
  exp335から継承した`guards.stage_s.saved_top1_value_parity_atol`をexp413 configから
  読んだため`KeyError: guards`で停止した。保存modelによる最終予測は0/75、
  学習boosterは0、回収可能predictionは0、submission生成/外部提出は0。
- 修正は同じbootstrapにSHA固定済みの`parent_exp335_config.yaml`からtoleranceを
  読む1箇所だけ。候補、scale5、seed、feature schema、保存model manifest、
  CPU runtime、zero-booster、submission禁止の契約は変更しない。
- 旧lookupがsourceにないことと親lookupがあることを専用回帰testへ追加した。
  同じ承認範囲でversion 2を再実行する。
- 修正後の専用＋親＋package helper testは`22 passed`。Jupytext
  round-trip、py_compile、Ruff `F821/F401/F811`、strict experiment
  validationもPASSした。
- version 2 bootstrap静的監査は50 support files、保存model 40/20/15、
  学習0、CPU/private/internet off、submission生成/submit falseでPASSした。
  embedded config / source / bootstrap済みNotebook SHA:
  `b4b8e9f148a89b81c11e380801c111ed4f90f707bac9e6a06714051320587bb1` /
  `425b3764b3eb13224c61c55a0bb2d6d964ebd45027c03856c2aee48daacff420` /
  `a6cdf200ae23f26f15625dc7a9636a3bfba7defca90d893bc0144aff0a9d915a`。

### 2026-07-29 current-test推論Kaggle version 2完了

- kernel version 2（id_no `128975306`）は`COMPLETE`。Notebook runtime
  `447.484 sec`、log最大timestamp `478.083568944 sec`。
- raw test 14,151 rows / 3 wellsを固定12候補、21 confidence列、
  selector88、nested74、signed23、clean273、final370の契約どおり再生成した。
- likelihood-PFは500 particles ×128 stable per-well seeds、temperature 5、
  gs×1.0。arithmetic meanとの差分は14,093/14,151 rows、scale5
  absolute/delta roundtrip、candidate formula、signed top1 parityはすべて0.0 ft。
- stable seedは`000d7d20=805188988`、`00bbac68=829597097`、
  `00e12e8b=1365511604`。arithmetic mean / scale5 content SHAは
  `b267952461cd2c54622a918adc775093a87cdd4c11087b05966bdf488fe74fd0` /
  `b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`。
- Stage C / S / Dの保存済み40 / 20 / 15 modelsを全件SHA検証して使用した。
  outer5 × config3の15 componentを等重み平均し、新規学習boosterは0。
- predictionはsample submissionの14,151 IDと順序が完全一致。ID重複、NaN、
  infは0。last-known + residual、15 component平均、feature group
  273/74/23を再監査してPASSした。
- prediction file / decompressed content SHA:
  `7f196d52994d604186f3d802c4743f4034d73eb79856b8ecee845aeab77f4048` /
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`。
- inference metrics / reproducibility manifest / downloaded log SHA:
  `40605fe82dc2ad06e836b9686aafc77ce8dd0ff048fd83d3a3ef58bea8b96e27` /
  `40605fe82dc2ad06e836b9686aafc77ce8dd0ff048fd83d3a3ef58bea8b96e27` /
  `bf132911bac364db97885c435200a9e8b36a2a7b91ce02856db98a4774aa0acd`。
- Kaggle output一覧と取得先を再帰監査し、`submission.csv`は0件。
  external submitもfalse。正規inference placeholderは変更していない。
- local `inference.run_enabled=false`、package `run_on_push=false`へ封印済み。
  submission生成と外部提出は別承認までfail-closedを維持する。
- 完了記録反映後の最終sealed packageも50 support files、run flag false、
  `run_on_push:false`、submission生成/submit falseをPASSした。embedded config /
  notebook / metadata SHA:
  `da01b6e8338196ee1232b1ec82f18d1bb2511727a63373a809eb6269f5f4feae` /
  `5a2ca05d75ba9606473e0bfa552e5e8b59fe446847e1979d6912d40c2743c46e` /
  `c5d602c8cb9b0ac6d7f2cf517d71d5135bb7788327e5dfe2678067d1a3eacd24`。

### 2026-07-29 submission.csv生成・submit-check

- ユーザーの`submission.csvを生成してください`により、検証済みpredictionから
  提出CSVを生成して事前検証する範囲を承認済みとした。外部提出は承認に含めない。
- source prediction file / decompressed SHA
  `7f196d52994d604186f3d802c4743f4034d73eb79856b8ecee845aeab77f4048` /
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`
  とsample SHA
  `7498f19ba1be281328c31a39044d4ba5f84e71c8f4115c613b5531f42aaff85a`
  を先に照合した。
- sampleの14,151 ID順へ`pred_tvt`をone-to-one strict joinし、列を
  `id,tvt`だけにして
  `kaggle/output/inference_v2/submission.csv`へ生成した。
- checkerと手動監査はPASS。14,151 rows、2 columns、unique ID 14,151、
  duplicate/missing/NaN/inf 0、sample header/row/order一致、source値exact parity。
- submission SHA / bytes:
  `e9bb6bca7e19a087997c1f8d1d708d8ba0af21e770f5e44e1f1a52078142772f` /
  `338475`。
- Kaggleへの外部提出、push、uploadは実行していない。
- submission生成完了記録を含むsealed packageはrun flag / `run_on_push` false、
  external submission authorization falseを再確認した。embedded config /
  notebook / metadata SHA:
  `22766e68640d7c61d90caf07665cda0d60adc31bc56e69e50036f8a2627ff199` /
  `877f098484d3ea4e4de43088ff6a7340bc55d2accd1be627b7913e3bf790dba5` /
  `c5d602c8cb9b0ac6d7f2cf517d71d5135bb7788327e5dfe2678067d1a3eacd24`。

### 2026-07-29 Kaggle submission outputへの訂正・version 3準備

- 上記`kaggle/output/inference_v2/submission.csv`は、取得済みversion 2
  predictionをローカルで変換した事前検証用CSVである。Kaggle Notebookの
  outputではないため、Code CompetitionへNotebook versionとして提出できる
  `submission.csv`ではなかった。これを最終提出候補と記録したのは誤り。
- ユーザー指摘に従い、同じfull current-test inference NotebookをKaggle
  version 3として再実行し、`/kaggle/working/submission.csv`をNotebook自身が
  生成する契約へ修正した。科学条件、12候補、stable seed、保存済み40/20/15
  models、CPU runtime、新規booster 0はversion 2から変更しない。
- version 3 output取得後に実ファイルをsample submissionへ照合し、行数、列順、
  ID順、重複、missing、NaN/inf、source prediction parity、SHAを検証する。
- Kaggle competition submit APIは呼ばない。外部提出authorizationはfalseのまま。
- version 3 push前検証は専用10件、package helper 4件、Jupytext round-trip、
  py_compile、Ruff `F821/F401/F811`、strict experiment validationをPASSした。
- bootstrap静的監査は50 support files、保存model 40/20/15、新規booster 0、
  current-test run / submission生成 true、external submit false、
  CPU/private/internet off、dataset source 0、kernel source 11を確認した。
- push前config / source / bootstrap済みNotebook / metadata SHA:
  `be8cf226861d868403b05da5fdd9ee8a19ebb2a078e46225ffbb28ff0df141d2` /
  `8cf48c0faa9391685f5a5de6f1f42118cd711f859804148914c043f5cc5fbace` /
  `205d672ce358e742aedab130e5f7b4e5d50ac4123b26e1d9b6862212e57dc49f` /
  `b906450354b928bb4fd8adcd99d901c5566fa7b073d00e1dbb28174b23d5216a`。

### 2026-07-29 Kaggle current-test inference version 3完了・submit-check

- kernel `kentookumura/exp413-scale5-likpf-current-test-inference`
  version 3（id_no `128975306`）は`COMPLETE`。Notebook内部runtime
  `306.297 sec`、log最大timestamp `329.69009102 sec`。
- Kaggle output一覧でrootの`submission.csv`を確認し、
  `kaggle/output/inference_v3_submission/`へ実ファイルを取得した。
- Kaggle生成`submission.csv`は14,151 rows、`id,tvt`の2 columns、
  unique ID 14,151。sampleとのheader / row count / ID順は完全一致し、
  duplicate / missing / NaN / infは0。version 3 predictionの`pred_tvt`と
  exact parity、最大絶対差`0.0 ft`でsubmit-checkをPASSした。
- submission SHA / bytes:
  `e9bb6bca7e19a087997c1f8d1d708d8ba0af21e770f5e44e1f1a52078142772f` /
  `338475`。
- version 3 prediction file / decompressed content SHA:
  `b9abab4aafeec6c4f5452167662ee5241be615136838e230a1af7c98cdd9f2b8` /
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`。
  decompressed contentはversion 2と同一。
- executed config / source / inference metrics / reproducibility / log SHA:
  `be8cf226861d868403b05da5fdd9ee8a19ebb2a078e46225ffbb28ff0df141d2` /
  `8cf48c0faa9391685f5a5de6f1f42118cd711f859804148914c043f5cc5fbace` /
  `14a758b19e8835ce082c503f798c0d8f8dcf0924f314b1613e588c7b9c48977d` /
  `14a758b19e8835ce082c503f798c0d8f8dcf0924f314b1613e588c7b9c48977d` /
  `1040174f5da418173e2ffa198bcc52bc4a395f25301321e56233ea09466d8988`。
- Kaggle competition submit APIは未実行。外部提出authorizationはfalse。
  local packageはpush直後から`run_enabled:false` / `run_on_push:false`へ封印済み。
- 最終検証は専用10 + 親exp335回帰8 + package helper 4 = `22 passed`、
  Jupytext round-trip、py_compile、Ruff `F821/F401/F811`、strict experiment
  validation、Kaggle実生成CSVのsubmit-checkをすべてPASSした。
- 完了記録を埋め込んだsealed packageは50 support files、`run_enabled:false`、
  `run_on_push:false`、submission生成契約true、external submit false、
  CPU/private/internet off。config / notebook / metadata SHA:
  `905c04d4e8a14674cd7272b17da23d6bbcc4c5aec16297e4699ad2de61128c6a` /
  `7ccfd9797c95299222be12625ac533e7457a64a02f13f685e928057d603b1b27` /
  `c5d602c8cb9b0ac6d7f2cf517d71d5135bb7788327e5dfe2678067d1a3eacd24`。

### 2026-07-29 code submission ref 55078306 hidden rerun失敗・version 4修正

- ユーザー実施のcode submission ref `55078306`、scriptVersionId
  `338788498`はCLI上`COMPLETE`だがPublic Scoreは空。raw APIの
  `errorDescription`は
  `Your notebook hit an unhandled error while rerunning your code. Note that
  the hidden dataset can be larger/smaller/different than the public dataset`。
- version 3 sourceはPF replay直後に公開commit test固有の14,151 rows / 3 wellsを
  hard assertしていた。hidden code rerun成功済み親exp335 ref `54928806`
  （Public LB `7.517`、errorDescription null）にはこのassertがない。
- 原因を`public_test_row_and_well_cardinality_hard_assert`と診断した。
  model、feature、seed、candidate、submission schemaの不一致ではない。
- version 4では科学条件を変えず、この2つの公開test固定assertだけを、
  `pf_frame`非empty、sampleとrow数・ID集合一致、ID一意、well数1以上という
  hidden-compatibleな動的契約へ置換する。
- 実行量は保存済み40 / 20 / 15 models、学習booster 0、CPU、internet off。
  外部competition submitは行わない。
- version 4 push前検証は専用10 + 親exp335回帰8 + package helper 4 =
  `22 passed`、Jupytext round-trip、py_compile、Ruff
  `F821/F401/F811`、strict experiment validationをPASSした。
- bootstrap静的監査は50 support files、dynamic sample row/ID contractあり、
  旧`current_test_expected_rows/wells`参照0、保存model 40/20/15、新規booster 0、
  run / submission生成 true、external submit false、CPU/private/internet off、
  kernel source 11。
- push前config / source / bootstrap済みNotebook / metadata SHA:
  `7b99d4533b86966e7004b1dd401cdd0f628dff755441aa15239b4e4cf0ffef03` /
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388` /
  `ab36121c34f20e45c3d9e1e6ee6bc637de6bfe5a1fbf58032108114b1400250a` /
  `b906450354b928bb4fd8adcd99d901c5566fa7b073d00e1dbb28174b23d5216a`。

## 2026-07-29 hidden互換current-test inference version 4完了

- kernel `kentookumura/exp413-scale5-likpf-current-test-inference`
  version 4（id_no `128975306`）は`KernelWorkerStatus.COMPLETE`。
- runtime `432.680 sec`、log max timestamp `462.24985278 sec`。
- 公開test固定cardinality assertだけを削除し、runtime契約を
  `len(pf_frame) == len(sample_submission)`、ID一意・集合一致、
  nonempty wellへ変更した。モデル、特徴量、candidate、PF seed、保存済み
  40 / 20 / 15 model、新規booster 0は変更していない。
- 14,151 rows / 3 wellsの公開outputではversion 3とprediction列、ID、
  `submission.csv`が完全一致し、全数値予測列のmax absolute differenceは
  `0.0`だった。
- Kaggle output
  `kaggle/output/inference_v4_hidden_compatible/submission.csv`は
  14,151 rows、`id,tvt`、unique ID 14,151、missing / nonfinite 0、
  sample順・source prediction parityを含むsubmit-checkをPASSした。
- executed config / source SHA:
  `7b99d4533b86966e7004b1dd401cdd0f628dff755441aa15239b4e4cf0ffef03` /
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388`。
- prediction gzip / decompressed SHA:
  `52ffb49110673f90b9b83b2e296e09b4ad0839164eda9ec13a91859937ebf136` /
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`。
- inference metrics / reproducibility manifest / log SHA:
  `8c26c5035c6b422738b351dd403674aa077bae5f838249f7e8ac5755943b3847` /
  `8c26c5035c6b422738b351dd403674aa077bae5f838249f7e8ac5755943b3847` /
  `3b4f2cbe105770adf5bbf2a64290062a667d87c19460c147c576b8551c600090`。
- submission SHA:
  `e9bb6bca7e19a087997c1f8d1d708d8ba0af21e770f5e44e1f1a52078142772f`。
- 実行後にrun flagを閉じて再生成したsealed packageのconfig / source /
  Notebook / metadata SHA:
  `c42d0c94202eba04b47b026200315afe0ef31295f74dc19392e8b9515c3e2cd4` /
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388` /
  `ba4db5ea97de230cc98aa7d593a5ca57a28b9900afa2a57b7e75503884bed3bf` /
  `c5d602c8cb9b0ac6d7f2cf517d71d5135bb7788327e5dfe2678067d1a3eacd24`。
- Codexからcompetition submitは実行していない。ユーザー側でversion 4を
  code submissionとして再提出する。

## 2026-07-30 version 4 code submission Public LB確定

- ユーザー実施のcode submission ref `55080377`をKaggle CLIで確認した。
  submitted at UTC `2026-07-29 11:29:12.383000`、
  `SubmissionStatus.COMPLETE`、Public LB `7.201`、Private LB未開示。
- kernelは`kentookumura/exp413-scale5-likpf-current-test-inference`
  version 4、提出fileは`submission.csv`。
- CV `7.884802794404715`、Public LB `7.201`。親exp335 Public LB `7.517`比
  `-0.316`、ensemble route anchor exp082 `7.601`比`-0.400`。
- exp413をML routeの新しいPublic-LB referenceへ更新する。ただしStage Dの
  by-well p95 `+1.228715 ft`、worst `+9.033462 ft`は変わらないため、
  LB reference更新とtrain-side robust promotionは分離する。
- Codexは外部submitを実行しておらず、ユーザー実施結果の確認・記録のみ行った。
- Public LB確定記録を反映して再生成したsealed packageは
  `run_enabled:false` / `run_on_push:false`を維持した。config / source /
  Notebook / metadata SHA:
  `d12e6d74a7f567f0873d5513883b3a7d36d0cd5be5231037e7db12f1a74036a7` /
  `0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388` /
  `e83b2233161bbe72353916afcc691be5998e00ae2d633a222a7fd05343604a94` /
  `c5d602c8cb9b0ac6d7f2cf517d71d5135bb7788327e5dfe2678067d1a3eacd24`。
