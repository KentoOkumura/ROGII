# exp239_distribution_matched_multicut_pseudotail セッションノート

## 目的

`distribution_matched_multicut_pseudotail`を実装し、exp023固定3-cutoffをofficial-start
分布に合わせたfold-safe deterministic samplerへ更新する。

## 現在の状態

- Route: `ml_model`
- 状態: v11 GPU full augmentation不採用、trial submission COMPLETE / Public LB 7.944 / 不採用
- 親: exp218 CV 8.475793752 / Public LB 7.843
- historical control: exp023 control 13.494554 / best 12.942938
- CV / LB: official-start OOF 8.697380066 / Public LB 7.944
- Kaggle train/inference/submit: CPU audit/cache v1-v10、GPU v11、inference v1、submission ref `54720769`すべてCOMPLETE

## GPUコストガード

- 最終active variant: `distribution_matched_multicut_weight050` 1
- LightGBM config: 3
- fold: 5
- booster: 15/15完走
- 親/control再学習: なし
- 保存済みexp218をcontrolとして参照し、親/control再学習は行っていない。

## v4 residual probe実行コスト

- active variant: `prefix_residual_learnability_probe` 1。
- LightGBM config: CPU deterministic 1。
- fold: source-well GroupKFold 5。
- booster: 合計5。early stoppingあり、最大500 rounds/booster。
- 親exp218/control再学習: なし。
- PF/Beam、learned likelihood、GRWR再生成: なし。
- synthetic pseudo-tail learnability probeであり、official-start OOFや採用判断の代替にはしない。

## 実装内容

- official-start well metadata: prefix/eval rows、prefix fraction、GR missingness、trajectory phase/tortuosity。
- cutoff sources: prefix/eval quantile、fixed hidden rows、GR robust change、GR missing boundary、trajectory curvature。
- deterministic matching: official-start各marginal bin deficitをglobal greedy quotaで充足し、source/hidden-like/new-wellは小さいsoft bonusに限定。
- caps: wellあたり0-3、source/wellあたり1、total 800、estimated augmentation ratio 0.45以下。
- fold-safe replay: source well 5-fold、request ID SHA、tail TVT feature read禁止、full-prefix cache slice禁止。
- v3 materialization: synthetic cutoff後を5距離帯から等間隔samplingし、残枠も決定的に補完して最大1,000 rows/request。
- v3 feature: anchor、prefix GR統計、recent 32/128 GR統計、row/anchor差分、XY/XYZ距離、dz/dMD。feature builderへTVT列を渡さずtargetを最後に付与。
- exp115 hidden-like role CSVをKaggle bootstrapへ同梱。
- inference notebookはno-submission contractのみ。

## 再現性

- cutoff生成とmatchingはglobal RNGを使わず、sorted wellとSHA256 stable tie keyで固定する。
- cutoff/fold/distribution/replay manifestのcontent SHAとschema SHAを保存する。
- gzipはdecompressed content SHAを主証拠とする。
- 現段階はmodel/prediction/submissionを生成しないためdeterministic prediction anchorではない。

## コマンドログ

- 2026-07-12: `task new-steering`はtask未導入のため失敗。
- 2026-07-12: `make new-steering EXP=exp239_distribution_matched_multicut_pseudotail`でsteering作成。
- 2026-07-12: `make new-exp EXP=exp239_distribution_matched_multicut_pseudotail`で実験作成。
- 2026-07-12: `py_compile`、`ruff check`、Jupytext変換と`--test`がpass。
- 2026-07-12: `make validate-exp EXP=exp239_distribution_matched_multicut_pseudotail`がstrict pass。
- 2026-07-12: canonical CPU train/inference Kaggle package prepareがpass。
- 2026-07-12: train bootstrap ZIP内のconfig cap/push guardとexp115 input同梱を確認。
- notebook local実行は行っていない。初回実行はKaggleとする。
- 2026-07-13: 53文字のinitial canonical slug `exp239-distribution-matched-multicut-pseudotail-train` は事前pull 403、push SaveKernel 400。Kaggle slug長制限の可能性を考慮し、ID/titleを同時に短縮した。
- 2026-07-13: `kentookumura/exp239-distribution-matched-multicut-train` v1をpush。COMPLETE、約48秒。
- 2026-07-13: outputを `/tmp/kaggle-output/exp239_distribution_matched_multicut_pseudotail/train_v1` に取得。

## Kaggle CPU audit v1

- 773 wells / official eval 3,783,989 rows。
- cutoff candidates 11,123、selected 1,546、replay requests 1,546、selected wells 773。
- estimated rows 1,545,558 / ratio 0.408447、row budget 1,702,795。
- source counts: fixed 704、quantile 607、GR change 181、missing boundary 43、curvature 11。
- hidden-like valid 204 wellsをすべてcoverage。
- request uniqueness、source-fold alignment、one-fold-per-well、min hidden 50、tail TVT feature read禁止、full-prefix cache slice禁止はpass。
- max marginal share delta 0.210220、mean abs delta 0.041871。
- `prefix_rows`最下位binはtarget 0.100906に対しselected 0.311125で+0.210220。
- decision: manifest/leakage contractはpassだがdistribution matchは要修正。exp218 feature regeneration / GPU trainへ進まない。

## 次

1. Kaggle CPU v3で800 replay requestsのprefix materialization row数、runtime、memory、SHAを監査する。
2. v3 guard通過後、learned likelihood / candidate rankerなど小さいdownstreamから段階評価する。
3. full exp218 GPU学習はvariant/config/fold/booster数を確定し、別途承認を得る。

## v2 global quota修正

- v1の全well一律2-cutoff round-robinを削除。
- global marginal deficitを全11,123候補で直接最小化し、0-3 cutoffs/wellを許可。
- hard caps: max 3/well、max 1/source/well、target/maximum total 800。
- soft weights: marginal 1.0、source 0.03、hidden-like 0.02、new-well coverage 0.03。
- 事前guard: max marginal差0.05以下、well coverage 0.65以上、hidden-like coverage 0.90以上、augmentation ratio 0.45以下、leakage全pass。
- v1 Kaggle候補/metadataのread-only preflight: 800 cutoffs、617 wells / 0.798189、hidden-like 204/204、augmentation ratio 0.211407、max marginal差0.030344、guard pass。
- preflightはselectorロジック確認でありKaggle v2結果ではない。
- Jupytext変換/test、ruff、py_compile、strict validation、canonical train/inference package prepareはpass。

## Kaggle CPU audit v2

- canonical kernel `kentookumura/exp239-distribution-matched-multicut-train` version 2、COMPLETE、約49秒、CPU。
- 773 wells / 11,123 candidatesから800 cutoffsを選択。
- selected wells 617 / 0.798189、hidden-like 204/204 / 1.0。
- estimated rows 799,961 / ratio 0.211407、cap 0.45以内。
- max marginal share delta 0.030344、mean abs delta 0.004408、guard 0.05以内。
- source counts: fixed 432、quantile 259、GR change 59、missing boundary 37、curvature 13。
- fold/leakage guardは全項目pass、LightGBM 0 config、booster 0、parent/control再学習なし。
- selected cutoff SHA `3eb8e1776387cc73596fd5faa53c2e76bd1dce14306570df41eb4e8e2625d6c6`。
- replay request SHA `710bc9e694ccb67c896a3f82657495dd61f2ed7999d336f5f91696f9fbfde26f`。
- output: `/tmp/kaggle-output/exp239_distribution_matched_multicut_pseudotail/train_v2`。

## v3 prefix materialization実装

- `anchor_and_prefix_statistics`だけを対象とし、PF/Beam、learned likelihood、GRWR、LightGBMは対象外。
- 5 distance bucketsへ各200行を割り当て、空き枠はtail全体から等間隔で補完。最大1,000 rows/request。
- `TVT`をdropしたframeだけをfeature builderへ渡し、生成後に`target_tvt`を別配列から追加する。
- request coverage、row cap、fold inheritance、target/feature分離、target finiteをhard assertion化。
- feature gzip、request summary、schema CSVとdecompressed content SHA / schema SHAを保存する。
- 2026-07-13: ruff format/check、py_compile、pure-function sampling smoke、Jupytext変換/test、F821、strict experiment validation、canonical Kaggle train/inference package prepareがpass。
- notebook local実行なし。
- 実行予定: active audit 1、LightGBM config 0、fold学習0、booster 0、parent/control再学習なし。

## Kaggle CPU audit v3

- canonical kernel `kentookumura/exp239-distribution-matched-multicut-train` version 3、COMPLETE、約96秒、CPU。
- 800/800 requests、799,961 rows、50 columns。requestあたりmin 961 / max 1,000 rows。
- materialized frame推定memory 528,026,392 bytes、gzip 68,757,754 bytes。
- request uniqueness、row cap、fold inheritance、target feature分離、target finiteは全pass。
- feature decompressed content SHA `cb6c7f401d88ecb9ac133d0ea035bbe626c54c2a7260aad62c7d0b4a989afa89`。download後の再hashも一致。
- feature schema SHA `56f31e7837b593b7d8c9c9a4d7a617c95aa5cba68ee7d4add962ff3429087cad`。
- request summary SHA `a9eff4fb331c9c45b564caa6f7e6e38764e06ac87b6f0a85d9a9399a53486461`。
- output: `/tmp/kaggle-output/exp239_distribution_matched_multicut_pseudotail/train_v3`。
- LightGBM 0 config、fold学習0、booster 0、parent/control再学習なし。

## v4 residual learnability probe実装

- v3の数値anchor/prefix/row geometryだけを使い、`target_tvt - anchor_tvt_input`を学習する。
- identifier、source well、fold、文字列bucket/source、targetをfeatureから除外する。
- source wellに割り当て済みの5 foldsをそのまま使い、派生requestを別foldへ出さない。
- baselineはanchor holdと`anchor_tvt_input + delta_z`。overall/distance bucket/by-wellを保存する。
- OOF coverage、one-fold-per-well、best baseline改善、最大well regressionをguard化する。
- active variant 1、LightGBM config 1、5 folds、合計5 boosters、parent/control再学習なし。

## Kaggle CPU audit v4

- canonical kernel version 4、COMPLETE、約212秒、CPU。
- 初回pushは他実験5 sessionがCPU枠を占有していたため上限エラー。別実験は停止せず、枠解放後に同じslugへpushした。
- 799,961 rows、44 numeric features、1 config x 5 folds = 5 boosters。
- fold RMSE: 17.772965 / 19.092937 / 19.680777 / 30.412258 / 31.693126。
- overall RMSE: anchor hold 69.526871、anchor + delta-z 156.505994、residual probe 24.349143。
- residual probeはbest baselineから-45.177728、全distance bucketでanchor holdを改善。
- 617 wells中407改善 / 210悪化。max well regressionは`86454a6f`の+63.415661でguard上限+20を超過。
- worst examples: `86454a6f` 42.424906 -> 105.840567、`89f1085d` 128.980125 -> 178.130883、`5305524b` 20.982182 -> 44.133550。
- top gain: prefix_fraction、delta_z、newly_hidden_rows、prefix_gr_mean、cutoff_index。
- OOF decompressed content SHA `a54f3b66d62f44a44ade8d7f5bd9e7f51006a9e0cb8cfedefd57baf36e388323`、download後再hash一致。
- decision: global learnabilityはpositiveだがworst-well guard failed。direct residual routeと親exp218学習へ進まない。利用を残すならcross-fitted confidenceまたはanchor shrinkageに限定する。

## v5 full exp218 augmentation本評価

- ユーザー承認: 全800 requests、pseudo weight 0.5、exp218全380特徴、3 configs x 5 folds = 15 boostersで本評価を進める。
- official rows 3,783,989はweight 1.0、pseudo rows 799,961はweight 0.5。実効augmentation mass約10.6%。
- validationはofficial-start rowsのみ。outer-valid source well由来pseudo rowsを対応foldのtrainから除外する。
- synthetic cutoffごとにexp072 base/PF/Beam/likPF、multi-observation、exp111/145 learned likelihood、U projection、exp218 GRWRを再生成する。full-prefix cache sliceは禁止。
- active variant 1、LightGBM config 3、fold 5、合計booster 15、parent/control再学習なし。
- 保存済みexp218 OOF 8.475793752を比較基準とする。
- `exp239_exp218_pseudotail_augmentation.py`を追加し、正規train notebookから上位orchestrationを呼ぶ既存notebook保守構成とした。
- ruff、py_compile、Jupytext test、F821、strict validation、bootstrap dependency manifest確認がpass。
- feature generationとGPU trainを単一Kaggle jobで行うため、過去実績exp072約4.3h + exp218約4.0hを踏まえ、12h上限リスクを監視する。
- 2026-07-13: GPU preflight v5は2 requestsのexp072/PF/Beam/likPF、learned likelihood、U projection、GRWR生成まで完走したが、pseudo側の基礎特徴選択がexp072 variantの除外規則を適用せず389列となり、期待380列guardで停止（約146秒）。LightGBM学習0、parent/control再学習なし。
- v5の原因に対し、pseudo側も`pixiux_likpf_public_replay`の基礎特徴選択を適用し、request管理3列を明示除外した。追加特徴の列単位代入を一括concatへ変更してfull evaluation時のDataFrame断片化を抑制した。
- 2026-07-13: GPU preflight v6は2 requests / 2,000 pseudo rowsで380 featuresを生成しpass。feature schema SHAは`1aff486413a6d86331d0f4537eea918dddc7db8d68cc8a0ea26a1a1297d777b9`、augmentation feature生成55.25秒、LightGBM学習0、parent/control再学習なし。
- v6 pass後にpreflightを解除。v7本評価はactive variant 1、LightGBM config 3、fold 5、合計15 boosters、全800 requests、pseudo weight 0.5、parent/control再学習なし。
- v7 GPU single-jobは開始約2時間56分（10,593秒）後、pseudo特徴生成中にPython tracebackなしの`DeadKernelError`で終了。時間上限ではなく、800 request DataFrameのlist保持と一括concatによるOOMが第一原因。LightGBM 0/15 boosters、official-start OOF未取得、parent/control再学習なし。
- ユーザー承認によりCPU feature-cacheとGPU trainingの二段階へ変更。CPU v8は25 requests/batch、予定32 Parquet shards、4 CPU jobs、800 requests / 799,961 rows / 380 features。config/fold/boosterは0/0/0でGPUを使わない。
- GPU v9はcache guard pass後のみ実行し、active variant 1、3 configs、5 folds、15 boosters、pseudo weight 0.5、parent/control再学習なしを維持する。
- CPU feature-cache kernel v1 preflightは25 requests / 25,000 rows / 380 features / 1 Parquet shardでpass。feature schema SHA `1aff486413a6d86331d0f4537eea918dddc7db8d68cc8a0ea26a1a1297d777b9`、生成434.50秒、peak RSS 5,885.05 MB、LightGBM 0 booster。
- v1確認後にcache preflightを解除し、同じCPU kernel v2で全800 requests / 予定32 shardsを実行する。
- CPU feature-cache kernel v2はCOMPLETE。32/32 shards、800 requests、799,961 rows、380 features、15,368.76秒（約4時間16分）、peak RSS 6,742.95 MB。feature columns SHA `1aff486413a6d86331d0f4537eea918dddc7db8d68cc8a0ea26a1a1297d777b9`、manifest SHA `ed9e5da6d979a6fa68f9f66ae86d46599393e43588ae535e1ad463b335d3501d`、schema SHA `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`、request manifest SHA `fb75b071e18b749498f525b56c0c721b8adfa22609d77251c156219d8acbfbd0`。
- GPU v9入力をCPU v2 outputに固定。active variant 1、3 configs、5 folds、15 boosters、official weight 1.0、pseudo weight 0.5、official-only validation、valid source-well pseudo除外、parent/control再学習なし。
- GPU cached-training kernel v1は32/32 pseudo shardsのfile SHA / row-content SHAと行数を全て検証・読込後、official 3,783,989行の380特徴assembly中に開始525秒で`DeadKernelError`。pseudo cache検証はpassしたが、pseudo全cacheをmemory保持したままofficial full frameを生成した順序によるOOMが第一原因。LightGBM 0/15 boosters、official-start OOF未取得、parent/control再学習なし。
- ユーザー承認により推奨の三段階構成へ変更。v10 CPU official cacheはexp218と同じ3,783,989 rows / 380 featuresを生成し、250,000 rows単位の16 Parquet shardsへ保存する。LightGBM 0 booster、GPU不使用。
- v10実装開始時のKaggle RUNNING sessionは4件で、pull metadata上すべて`enable_gpu=false`。ユーザー指定のCPU session 5件ではないため、push直前に再確認して4件以下なら実行する。
- v10 package完成後の再確認では既知CPU RUNNING 4件だったが、push直前に別`[Private Notebook]`が開始し、Kaggleが`Maximum batch CPU session count of 5 reached`で拒否。v10は未開始。再確認でも既知4件+private 1件の計5件のため、ユーザー指示どおり再pushせずCPU枠待機。
- 2026-07-14再実行時はRUNNING 3件を個別status確認し、3件ともpull metadata上`enable_gpu=false`のCPU sessionだった。5件未満のためv10 pushを再開した。
- 元slug `kentookumura/exp239-official-exp218-feature-cache` はCLI 2.2.0/2.2.2とも`Notebook not found`で作成できず、pullは500、検索結果にも既存実体なし。package metadataに誤混入していた不要なexp099/111/112 sourceをconfigどおり除外しても同じだったため、前回CPU上限拒否時の不完全なslug予約と判断した。
- 学習条件・入力・出力を変えず、復旧slug `kentookumura/exp239-official-exp218-feature-cache-v1`へ変更。Kaggle kernel version 1のpushに成功し、status `RUNNING`を確認。予定はofficial 3,783,989 rows / 380 features / 16 Parquet shards、LightGBM 0 booster、GPU不使用。
- v10 official feature cache kernel v1はCOMPLETE。16/16 shards、773 wells、3,783,989 rows、380 features、745.44秒（約12分25秒）、peak RSS 29,394.73 MB。feature columns SHA `1aff486413a6d86331d0f4537eea918dddc7db8d68cc8a0ea26a1a1297d777b9`、manifest SHA `a2c89e0ae432ac59e8f8ffee457f530398707549d9b87eded44d6cb5ac0b27ef`、schema SHA `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`。pseudo cacheとfeature/schema SHAが一致。LightGBM 0 booster、parent/control再学習なし。
- 次段階v11はofficial 16 shardsとpseudo 32 shardsを一括DataFrame化せず、disk-backed arrayへ順次読み込んでfold学習する。active variant 1、3 configs、5 folds、15 boosters、official weight 1.0、pseudo weight 0.5、official-only validation、valid source-well pseudo除外、parent/control再学習なしを維持する。
- v11実装ではofficial/pseudo各shardのfile SHA・row-content SHA・行数を検証しながらfloat32 memmapへ書く。outer-valid wellはint codeで照合してpseudoから除外する。各foldのtrain/valid行列も一時memmapとし、booster学習後に削除する。成功時はbase cache memmapも閉じて削除し、Kaggle outputへの巨大一時ファイル混入を防ぐ。
- 既存v9 notebookは失敗履歴として上書きせず、`exp239_distribution_matched_multicut_pseudotail_dual_cache_streaming_train.ipynb`を別名で作成。py_compile、ruff F821/E9、Jupytext test、strict experiment validation、strict Kaggle package prepareがpass。
- v11 GPU実行予定を再確認: active variant 1、LightGBM config 3、fold 5、合計15 boosters、parent/control再学習なし。保存済みexp218 OOF 8.475793752をcontrolとする。kernel idは`kentookumura/exp239-pseudotail-dual-cache-streaming-train`。
- v11 kernel version 1をpushし、status `RUNNING`を確認。pull metadata上`enable_gpu=true` / `machine_shape=Gpu`、入力はpseudo cacheとofficial cache v1の2件のみ。ユーザー指定どおり継続監視はせず、完了連絡後にlogsからOOF・fold score・SHA・peak RSSを回収する。
- v11 kernel version 1はCOMPLETE。48 cache shardsを全て検証し、3 configs x 5 folds = 15/15 boostersを完走。official-start OOF RMSEは8.697380066で、保存済みexp218 8.475793752から+0.221586314（約+2.61%）悪化した。実装/schema guardはpassしているためpseudo-tail追加仮説のnegative resultと判断し、inference/submitは行わない。
- 2026-07-15: ユーザー明示依頼によりnegative CVを承知でv12 trial submissionへ進む。v11 outputの15/15 modelを取得し、全model SHAと380-feature schemaを検証。exp218 raw-test replayへv11 boosterのみ差し替えるinferenceを実装し、Jupytext test、py_compile、ruff F821/F401/E402、strict experiment validationがpass。
- 初回inference push `kentookumura/exp239-distribution-matched-multicut-pseudotail-inference` は、57文字slugに対するKaggle `SaveKernel 400`で実行前に失敗。同idのpullは403で存在を確認できず。意味を保つ46文字canonical slug `kentookumura/exp239-distribution-matched-multicut-inference`へid/titleを同時に揃えて再prepareする。
- 短縮canonical inference kernel v1はpush/実行ともにCOMPLETE。約134秒、GPU、internet無効。v11 model 15/15 SHA、ordered 380-feature schema、test 14,151 rows、fallback 0を確認。submission SHAは`81d16997fe50f5e89186906d1fa5f1d70d255b4abc452ff91950fa1b59d5ccee`。
- `/tmp/kaggle-output/exp239-inference-v1/submission.csv`を取得し、sample submissionに対するcheckerはFAIL/WARNなしでPASS。header/行数/ID順完全一致、重複0、NaN/Inf 0。
- code submission ref `54720769`。message=`exp239 v11 pseudo-tail trial; CV 8.697380; submit-check PASS`、2026-07-15 10:00:44 UTC時点PENDING。過去のユーザー指定に従い連続monitorは起動せず、確定後にPublic LBを追記する。negative OOFによる不採用判断は維持する。
- 2026-07-15 scoring完了。ref `54720769`は`SubmissionStatus.COMPLETE`、Public LB 7.944、Private LB未表示。exp218 7.843より+0.101、exp238 ML anchor 7.775より+0.169、exp082 ensemble anchor 7.601より+0.343悪化した。official-start OOF 8.697380066の悪化方向とLBが一致したため、直接pseudo-tail row混合は不採用を確定し、weight微調整や同方式の追加提出は行わない。
- runtime 15,371.68秒（約4時間16分）、peak RSS 19,509.44 MB。prediction decompressed content SHA `16f77eccfb66d6c702ed2b70b33dfedb7544a51e7b00877bff8093371fe17ce9`、pseudo manifest SHA `ed9e5da6d979a6fa68f9f66ae86d46599393e43588ae535e1ad463b335d3501d`、official manifest SHA `a2c89e0ae432ac59e8f8ffee457f530398707549d9b87eded44d6cb5ac0b27ef`、feature/schema SHA `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`。parent/control再学習なし。
