# exp244_bidirectional_prediction_start_pseudotail_augmentation セッションノート

## 目的

`bidirectional_prediction_start_pseudotail_augmentation` backlogを実装する。exp239の
early-start replay contractへofficial controlとtrain-only late-startを追加し、同じsource-well
foldを継承するearly/original/late multi-view manifestとprefix再生成監査を作る。

## 現在の状態

- Route: `ml_model`
- 状態: v3 dual-start confidence-shrink Kaggle CPU評価完了、不採用
- 親: `exp239_distribution_matched_multicut_pseudotail`
- model比較親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- official-start OOF: raw `8.475793978` / v3 `8.477243182`（悪化、未採用）
- Public / Private LB: 未提出
- blocked: none

## 変更点 / 実装内容

- 固定start offset `-1000/-250/0/+250/+1000` rowsをearly/original/lateへ分類する。
- prefix 200 rows以上、remaining tail 50 rows以上、最大5 views/well、最大4,000 viewsをguardする。
- late viewは`target_usage=train_only_augmentation`、`current_test_compatible=false`へ固定する。
- source well GroupKFoldを全派生viewへ継承し、outer-train exclusion keyをsource wellに固定する。
- raw horizontal frameからstartごとに`TVT_input`を作り直し、start後のtrue TVTをfeature frameへ渡さない。
- anchor、prefix GR、trajectory、distance特徴をdistance bucketから最大1,000 rows/viewで決定的にsampleする。
- manifest、fold、distribution、request summary、feature schema/content SHAを保存する。
- test-sideはnegative offsetだけを使い、pseudo start後からactual startまでのknown prefixだけをcalibration backtestにする。
- test-sideでactual start超過、unknown tail TVT、full-model fine-tune、submission predictionをhard assertionで禁止する。
- PF/Beam、learned likelihood、GRWR、PF/HMM初期状態は初版では生成せず、後続再生成contractのみ保存する。

## 学習コスト確認

- active audit: 1
- LightGBM config: 0
- fold学習: 0
- 合計booster: 0
- parent/control再学習: なし
- GPU: 不使用

## 実行・検証ログ

```bash
make new-steering EXP=exp244_bidirectional_prediction_start_pseudotail_augmentation
make new-exp EXP=exp244_bidirectional_prediction_start_pseudotail_augmentation
.venv/bin/ruff format <train.py> <inference.py>
.venv/bin/python -m py_compile <train.py> <inference.py>
.venv/bin/ruff check <train.py> <inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <train.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <inference.py>
.venv/bin/ruff check <train.py> <inference.py> --select F821
make validate-exp EXP=exp244_bidirectional_prediction_start_pseudotail_augmentation
make prepare-kaggle-notebooks EXP=exp244_bidirectional_prediction_start_pseudotail_augmentation EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp244-bidirectional-pseudotail-augmentation-train --title 'exp244 bidirectional pseudotail augmentation train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp244_bidirectional_prediction_start_pseudotail_augmentation EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp244-bidirectional-pseudotail-augmentation-inference --title 'exp244 bidirectional pseudotail augmentation inference' --run-on-push --strict"
make validate-template
make test
```

結果:

- ruff format/check: PASS
- py_compile: PASS
- Jupytext conversion/test: PASS
- F821: PASS
- strict experiment validation: PASS
- train/inference Kaggle package prepare: PASS
- template validation: PASS
- pytest: 13 passed
- metadata: private、CPU、internet off、competition sourceあり、run-on-push true
- bootstrap config: offset、active audit、config/fold/booster、parent-control、submission guardが正本と一致
- notebook local実行: 未実行（Kaggle-firstルールを維持）
- Kaggle push: 未実行

## Notebook構造確認

- 親exp239に`*compact_selfcontained*_train.py`は存在しないため、正規train.pyを構成参照元とした。
- exp239 train.py 1,497行に対し、exp244 train.py 778行、inference.py 471行。
- train notebookは8章、inference notebookは6章で、入力解決、manifest生成、prefix再構成、guard、保存をセル上で追える。
- 同一exp helper importは使わずself-containedとした。
- train/inference notebook sourceに`__file__`は残っていない。

## 再現性メモ

- view ID / fold key: experiment、source well、official start、offsetからSHA256で生成する。
- stochastic components: なし。
- ordering: well/view/requestをstable sortする。
- gzip: raw gzip SHAは非主証拠、canonical decompressed CSV content SHAを主証拠とする。
- deterministic anchor: Kaggle rerun SHAが未取得なので、現時点では主張しない。
- model / prediction / submission SHA: 対象生成物なし。

## 次のアクション

1. Kaggle CPU auditを実行する場合は、active audit 1、LightGBM 0 config、fold学習0、booster 0、parent/control再学習なしを再確認してpushする。
2. train manifest/materialization guard通過後にだけ、learned likelihood / ranker / small calibratorのどれを最初のdownstreamにするか決める。
3. official-start OOFを得るまでは推論予測・提出へ進まない。

## 2026-07-14 v2 frozen-anchor parity preflight

### 事前監査で判明したfold不一致

- exp218 train v1 outputを`/tmp/exp218_train_v1_output`へ取得し、後続入力に必要なOOF、model manifest、15 saved boostersを確認した。
- frozen OOF: 3,783,989 rows / 773 wells / `lgb_mean` RMSE 8.475793978（記録値8.475793752との差は丸め範囲）。
- OOF decompressed SHA: `5f3fc95182eea348f3545771e67778ce191e7ba468eee7b267f4993369422976`。
- model manifest SHA: `904570def0d6ad0140f3df95c8bb38f31823295fd191206290e3833b5b2cc237`。
- exp218は3,783,989 row-level groupsのGroupKFold、exp244 v1は773 unique wellsを等weightにしたGroupKFoldだった。
- ローカルでfold規則を再構成した参考値は155 / 773 wells一致、618 wells不一致だった。ただし等weight tie順序がNumPy環境に依存するため、後述のKaggle v2実manifest比較で置き換えた。
- v2ではraw official-tail row数をweightにしてsklearn non-shuffled GroupKFoldと同じ割当を再構成する。

### v2実装

- `exp244_bidirectional_prediction_start_pseudotail_augmentation_guard.py`を追加。
- exp218 OOFのexperiment/variant/mode/model、行数、well数、RMSE、decompressed SHAをhard assertionする。
- raw trainのofficial-tail行数・first/last OOF IDとexp218 OOFをwell単位で完全照合する。
- exp218 model manifest、15 booster、fold 0..4、全model file存在を検証する。
- v1 / v2 fold、変更well、fold別row数、canonical content SHAを保存する。
- 既存v1 train notebookは上書きせず、fold修正版を別名`*_v2_train.ipynb`へ生成した。

### 学習コスト確認

- active audit: 1
- LightGBM config: 0
- fold学習: 0
- 合計booster: 0
- parent/control再学習: なし
- frozen exp218再推論: なし
- GPU: 不使用
- prediction / submission: なし

### 静的検証

- ruff format/check: PASS
- py_compile: PASS
- Jupytext conversion/test: PASS
- strict experiment validation: PASS

### Kaggle guard v1 / v2

- canonical kernel: `kentookumura/exp244-frozen-anchor-parity-guard`。
- v1はexp218 OOF/raw surface/model manifestの全guard pass。ただしv1 foldを同じ規則から再構成したため、tie順序依存が残った。
- v2はexp244 train v1の実fold manifestをkernel sourceとして追加し、SHA `e896c1afbba8d6cefe29f52d3073a2265c184ceb992bbe892654752f47f28176`を照合して直接比較した。
- v2 result: COMPLETE / `frozen_anchor_parity_preflight_passed`。
- runtime: Kaggle CPU、reported elapsed 38.651 sec、internet off、GPUなし。
- frozen exp218 OOF: 3,783,989 rows / 773 wells / RMSE 8.475793978195085。
- OOF decompressed SHA: `5f3fc95182eea348f3545771e67778ce191e7ba468eee7b267f4993369422976`。
- model manifest: 15 boosters、fold 0..4、全model fileあり、SHA `904570def0d6ad0140f3df95c8bb38f31823295fd191206290e3833b5b2cc237`。
- raw official surfaceはwell、tail行数、first/last OOF indexまで完全一致。
- v1 foldとv2 exp218互換foldは174 wells一致、599 wells変更、match rate 0.225097。
- v2 fold rows: fold0 757,738 / fold1 756,650 / fold2 756,255 / fold3 757,101 / fold4 756,245。
- parity fold manifest SHA: `33694a894b4c616b8ac187d9cb752171c70815a5e607c3c27beccfb896883c0f`。
- fold report SHA: `3b3ee689fe3a3324de0d1b7b8aebca896f1360d268440f519df40ecf0633ca3c`。
- model training / inference prediction / submission: すべてなし。
- 後続入力に必要なfold manifestとsummaryを`kaggle/output/guard_v2/`へ取得した。

### v2判断

frozen exp218のidentityとofficial-start評価面は再利用可能。exp244 v1 foldはleakage-freeだがexp218との
outer-fold比較には不適切なので廃止し、後続はv2 parity fold manifestへ固定する。次はこのfold上で
single-variantのconfidence-shrink meta-validationを設計し、official-start OOFでのみ採否を決める。

## 2026-07-14 Kaggle CPU train audit v1

- 実行前確認: active audit 1、LightGBM config 0、fold学習0、booster 0、parent/control再学習なし。
- runtime: CPU、internet off、private、competition sourceあり。
- canonical kernel: `kentookumura/exp244-bidirectional-pseudotail-augmentation-train`
- version: 1
- push: 成功、実行開始。
- URL: https://www.kaggle.com/code/kentookumura/exp244-bidirectional-pseudotail-augmentation-train
- result: COMPLETE。
- wells / views / rows: 773 / 3,854 / 3,850,880。
- view counts: early 1,537、original 773、late 1,544。
- view share: early 0.398806、original 0.200571、late 0.400623。late上限0.45内。
- reported audit elapsed: 61.299717 sec。Kaggle notebook log終端は約445 sec。
- guards: all pass。original control全well、source-fold alignment、late train-only、unknown-tail禁止、full-prefix cache slice禁止、tail TVT feature read禁止、request coverage、target finite。
- feature decompressed content SHA: `3ad67ca37800b28e6a77f8a25fbbf8167dbe60bfbaa7922ab19c03847706b444`。
- feature schema SHA: `19631f7c8e7a7cfcfbe36f698fa53e7ba0f2d1508cf328ee71fdeb74bf627d24`。
- view manifest SHA: `2969890a3959b07070c8af3221c23ca66e84c3de1fa1660e1e9172499f93a028`。
- fold manifest SHA: `e896c1afbba8d6cefe29f52d3073a2265c184ceb992bbe892654752f47f28176`。
- distribution SHA: `c1b6b3ba6980fa8f90507b4a3109b90c59e6925b783b32d1103b1855108cad7d`。
- model training / inference prediction / submission: すべてなし。
- output archive: 未取得。logsにguard、件数、SHA、生成物パスが揃っており、現段階の評価には不要。

## 2026-07-14 Kaggle inference audit push復旧

- 初回ID: `kentookumura/exp244-bidirectional-pseudotail-augmentation-inference`。
- 初回push: 詳細なし`SaveKernel 400`。Kaggle側にnotebookは作成されず、pullは403、mine listにも存在しない。
- 原因: slugが54文字で、成功したtrain slug 50文字より長くKaggle上限を超えた可能性が高い。
- 復旧ID: `kentookumura/exp244-bidirectional-pseudotail-inference`。
- title: `exp244 bidirectional pseudotail inference`。ID/title slugを一致させる。
- 実験番号と実験ディレクトリは変更しない。
- 復旧push: 成功、version 1として実行。
- URL: https://www.kaggle.com/code/kentookumura/exp244-bidirectional-pseudotail-inference
- result: COMPLETE。
- test wells / requests / rows: 3 / 6 / 3,750。
- offsets: 各well `-1000/-250`。known calibration rowsは1,000/250。
- reported elapsed: 2.251276 sec。
- guards: request ID unique、early known-prefix only、actual start非超過、全materialized row既知prefix内、unknown-tail TVT禁止、full-model fine-tune禁止、submission prediction禁止。すべてpass。
- calibration feature content SHA: `6cce096380821bf7df2ab6ba0a22b11d31ff6808db57f810a21014e1e7370fdd`。
- calibration feature schema SHA: `9a4af22feb32f58654dc6da177d897d9a6dd0f7d6c53b2d1db9d1136f6a7b034`。
- calibration request manifest SHA: `37825f74fb2148740f4e169b5efbe8a1188ad7309270a92fe52c4c291f41c588`。
- prediction / submission: 生成なし。
- output archive: 未取得。logsに件数、guard、SHA、生成物パスが揃っているため不要。

## 2026-07-14 v3 dual-start confidence-shrink meta-validation実装

### 設計判断

- exp218 OOFからouter-train optimal alphaを学習するmeta modelは、nested base OOFがなくfold attributionが曖昧になるため採用しない。
- 各well自身のknown prefixだけで`-1000/-250` startのlocal-linear backtestを行う。
- 各pseudo start以前の最後128 rowsでTVT対MDをfitし、pseudo start後からofficial startまでの既知区間でRMSEを測る。
- 2 start RMSEのminimumが10 ft以下ならalpha=1、10〜30 ftをlinear ramp、30 ft以上をalpha=0.95とする。
- `pred = anchor + alpha * (exp218_pred - anchor)`だけをofficial tailへ適用する。
- formula、threshold、max shrinkはofficial-tail truth、fold結果、他well truthからfitしないsingle pre-registered variant。

### 入力

- exp218 frozen `lgb_mean` OOF。decompressed SHA `5f3fc951...2976`を要求。
- v2 parity fold manifest。SHA `33694a89...c0f`を要求。
- exp115 hidden-like fold assignment。SHA `5f9ac9fa...6597`をbootstrap dependencyとして同梱。
- raw competition train。calibration featureはrawから再生成し、official startを超えてTVT_inputを読まない。

### 学習コスト確認

- active variant: 1
- LightGBM config: 0
- fold学習: 0
- 合計booster: 0
- parent/control再学習: なし
- alpha学習model: なし
- GPU: 不使用
- test prediction / submission: なし

### 実装

- `exp244_bidirectional_prediction_start_pseudotail_augmentation_train_variant0.py`を追加。
- overall / 6 distance buckets / 1000+ / exp115 spatial / typewell-purged / 5 folds / by-wellを評価する。
- worst-well +2 ft、hidden-like非悪化、1000+非悪化、3 folds以上改善をadoption guardにした。
- calibration features、metrics、fold metrics、by-well、OOF、summaryとcanonical SHAを保存する。

### Kaggle CPU v3結果

- kernel: `kentookumura/exp244-dual-start-confidence-shrink` v1。
- status: COMPLETE / `confidence_shrink_meta_validation_complete`。
- runtime: CPU、internet off、reported elapsed 180.541 sec。
- raw exp218 OOF: 8.475793978195084。
- calibrated OOF: 8.477243182127168。
- delta: +0.001449203932（悪化）。
- eligible dual-start wells: 772 / 773、shrink使用33 wells、全well使用率4.269%。
- alpha min / mean / max: 0.95 / 0.998726208 / 1.0。
- start RMSE gap median 273.454859、p95 385.685030で、`-1000`と`-250`のlinear proxyは大きく不一致。
- bucket: 000_050 -0.000484、050_100 -0.001645は改善したが、100_250 +0.004951、250_500 +0.005790、500_1000 +0.004353、1000_plus +0.001200で悪化。
- hidden-like spatial +0.000603、typewell-purged +0.000602で両方悪化。
- fold delta: +0.003354 / -0.002405 / +0.000170 / +0.001412 / +0.004399。改善1 / 5 folds。
- worst-well regression: `a959858c` +0.925075で+2 ft guard内。
- 使用33 wellsのうち15改善 / 18悪化、delta median +0.004423、mean +0.039585。
- dual-start riskとwell deltaの相関は-0.0181で、proxy confidenceとしてほぼ無相関。
- adoption guards: overall、1000+、hidden-like 2面、minimum foldsがFAIL。worst-wellのみPASS。`adoption_supported=false`。

### SHA / 生成物

- calibration feature SHA: `96dc811153e31af02b7778cd344852a05c1204073ac321e1a44f58dd09f1571a`。
- metrics SHA: `e4df91ca7c9b7076a7beefb9193aa7896c723b472965af026928b4469a68fdb8`。
- fold metrics SHA: `b340f1a89739440fabd1a4bb28b306b8d5bc4c05992bae20bde2f85ab6ae38d7`。
- by-well SHA: `b89b62ea79d7c27484d1b492be5d5b1e8800f68630550642b3ca1fed2de95da2`。
- OOF decompressed SHA: `e55fec0351f2eaef727ffdb47ce5a296633ead2a152ea606017871564646c747`。
- output: `kaggle/output/confidence_shrink_v1/`。
- model training / test prediction / submission: すべてなし。

### v3判断

known-prefix local-linear backtest errorはexp218 residual confidenceを識別せず、33-wellの小さな使用率でも
official-start long-tailへ転移しなかった。threshold / max-shrink grid、current-test port、submissionは行わない。
再開条件は、exp218自身をpseudo startで再生成したprediction backtestがoriginal-start parityを通り、
local-linear proxyとは独立したconfidence evidenceを示すこと。優先度は低-中へ下げる。

## 2026-07-14 v4 本来のearly / original / late統合学習を実装

### v3との切り分け

- v3のlocal-linear confidence shrinkはtest-time calibrator枝であり、prediction start位置を変えた
  dataをexp218-family modelへ学習させていなかった。
- v4は中心仮説を直接検証する。original全行と4 offsetのearly/late rowを同じLightGBM学習へ入れる。
- 保存済みexp218 `lgb_mean` OOFをcontrolとし、control/親実験は再学習しない。

### 学習データ契約

- original: exp239 official 380-feature cache、3,783,989 rows、weight 1.0。
- `-1000`: 764 views / 191,000 rows。
- `-250`: 773 views / 193,250 rows。
- `+250`: 773 views / 193,157 rows。
- `+1000`: 771 views / 192,750 rows。
- pseudo合計: 3,081 views / 770,157 rows、weight 0.5。
- 各viewは距離`0-49 / 50-249 / 250-999 / 1000-2499 / 2500+`から各50行、
  最大250行を決定的にsamplingする。
- 4 offsetともraw horizontalからstart位置に応じた`TVT_input`を作り直し、exp072/PF/likPF、
  learned likelihood、U projection、GRWRを含む380特徴を再生成する。full-prefix cache sliceは使わない。
- late viewの追加true TVTはtrain-only。outer-valid source well由来pseudo rowは当該foldのtrainから除外する。
- validationはofficial-start全3,783,989 rowsだけに固定する。

### 学習コストと承認guard

- active variant: 1 (`bidirectional_balanced250_weight050`)
- LightGBM configs: 3
- folds: 5
- 合計boosters: 15
- parent/control再学習: なし
- feature-cache: CPU 4 jobs / 0 booster
- inference / submission: 無効
- `model.integrated_augmentation.run_approved=false`。4 cacheのSHA契約通過後、GPU push前に
  上記15 boostersを提示して明示承認を得るまで実行しない。

### 実装

- offset別Jupytext notebookを4本追加:
  `..._multiview_cache_m1000/m250/p250/p1000.{py,ipynb}`。
- `..._integrated_train.{py,ipynb}`を追加。official/pseudo cacheのmanifest/schema/file/row SHAを検証し、
  disk-backed memmapへstreamする。
- exp218の3 configを5-foldで学習し、model/importance/schema/OOF/metrics/by-well/summaryとSHAを保存する。
- overall、6 distance surfaces、1000+、hidden-like spatial/typewell-purged、5 folds、worst-wellを
  frozen exp218 OOFと同じofficial surfaceで比較する。
- adoption guardはoverall改善、1000+非悪化、hidden-like 2面非悪化、worst-well +2 ft以内、
  3 / 5 folds以上改善。
- cache 4 kernelとGPU train kernelのstrict Kaggle packageをprepareした。pushは行っていない。

### 静的・契約検証

- raw train 773 wells / official 3,783,989 rowsを読み、4 offsetのview/row数が全設定値と一致。
- Jupytext変換と`--test`: 5 notebooks pass。
- ruff、py_compile、strict experiment validation、template validation: pass。
- repository pytest: 15 passed。
- prepared metadata: cache 4本はCPU/internet off、integrated trainはGPU/internet off。
- integrated train inputs: exp239 official cache、4 offset cache、exp218 train artifact。
- cache/GPU実行、model生成、test prediction、submission: すべて未実施。

## 2026-07-15 v4実行承認・offset cache開始

- ユーザーへ新規1 variant / LightGBM 3 configs / 5 folds / 15 boosters、parent/control再学習なしを提示済み。
- その提示に対する「実行してください」をGPU学習を含む明示承認として記録する。
- 実行順は4 CPU offset feature-cache（各0 booster）を先に行い、全manifest/schema/request/row/file SHA
  契約が通った場合だけGPU学習へ進む。
- cache検証中は`model.integrated_augmentation.run_approved=false`を維持する。
- GPU実行対象は`bidirectional_balanced250_weight050`のみ。保存済みexp218 OOFをcontrolとし再学習しない。
- inference prediction / submissionは今回の実行範囲外。

### CPU cache push

- `kentookumura/exp244-multiview-cache-m1000` v1: push成功、CPU実行開始。
- `kentookumura/exp244-multiview-cache-m250` v1: push成功、CPU実行開始。
- `kentookumura/exp244-multiview-cache-p250` v1: push成功、CPU実行開始。
- `kentookumura/exp244-multiview-cache-p1000` v1: push成功、CPU実行開始。
- 4 kernelともprivate、CPU、internet off、run-on-push true、competition sourceと5 parent kernel sourcesあり。
- title slugとkernel IDは一致。別slugや再pushは行わず同じv1を監視する。

### 監視停止

- 2026-07-15、4 kernelがすべて`KernelWorkerStatus.RUNNING`であることを確認。
- ユーザー指示によりローカルの60秒pollingだけ停止。Kaggle上のv1実行は停止していない。
- 完了連絡後、同じkernel IDの通常logsを取得し、件数・380-feature schema・manifest/request/row/file
  SHA契約を確認する。4本すべて通過するまでGPU trainは開始しない。

### CPU cache v1完了・GPU gate通過

- 4 kernelとも`KernelWorkerStatus.COMPLETE`。Traceback / Error / Exceptionなし。
- `m1000`: 764 requests / 191,000 rows / 31 shards、13,980.001 sec、peak RSS 7,969.14 MB。
  manifest SHA `1471fa08662b5af1e5870ac3ea74ee3f301acad5659645e07552f3e1c574f289`、
  request SHA `b16c057d015062a1a4fe2a4d888cf375fcac0d8421d08e08a836f13fd699e005`。
- `m250`: 773 / 193,250 / 31、13,276.436 sec、peak RSS 6,455.00 MB。
  manifest SHA `a30f23b78bb65b3fca59f7366cf464b6b5cb86ce3f90ffce98115acb221e7923`、
  request SHA `6ac53c5aeaf7014a0fd06df40202bfa0386d16a1af493480f18a3d3a381690e2`。
- `p250`: 773 / 193,157 / 31、11,856.186 sec、peak RSS 6,858.81 MB。
  manifest SHA `6c40ab4a3eef6e7b53c699764f8476db375faf9eba93ec4fbab3011a38a4f979`、
  request SHA `cabf50b89a3191189ac78084b1d4a445b400d713d32c0455a4b41723c85cd135`。
- `p1000`: 771 / 192,750 / 31、11,996.211 sec、peak RSS 5,318.82 MB。
  manifest SHA `4262c3b218ad4073a8813f88a7fc8d74d3cd95cc7ccf0832da03cf9c3e1611c1`、
  request SHA `9d9c144830ba15353fc78f14f9abe30b78c24773665b37aaf3a328d3a973c2ac`。
- 全4本のfeatures=380、feature-columns SHA
  `1aff486413a6d86331d0f4537eea918dddc7db8d68cc8a0ea26a1a1297d777b9`、schema SHA
  `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`が一致。
- preflight=false、full-prefix cache slice禁止、tail TVT feature read禁止。`p250/p1000`だけ
  `late_train_only=true`、early 2本はfalse。
- 合計3,081 requests / 770,157 rows / 124 shards。各manifest SHAがper-shard file/row SHAを固定する。
- 上記v1 SHAを`config.yaml`へpinし、GPU notebookは124 shardsを実読込・再hashしてから学習する。
- cache prerequisite通過につき`run_approved=true`へ変更。実行量は承認済みの1 variant / 3 configs /
  5 folds / 15 boosters、parent/control再学習なし。

### GPU integrated train push

- kernel: `kentookumura/exp244-bidirectional-multiview-train` v1。
- URL: https://www.kaggle.com/code/kentookumura/exp244-bidirectional-multiview-train
- push成功、run-on-pushで実行開始。
- private、GPU enabled、internet off、competition sourceあり。
- kernel sources: exp239 official cache、exp244 offset cache 4本、exp218 train artifactの計6本。
- active variant 1 / LightGBM 3 configs / 5 folds / 15 boosters。parent/control再学習なし。
- inference / submission生成なし。official-start OOFとadoption guardsだけを評価する。
- push後pull成功、Kaggle id_no `127301177`、machine shape `Gpu`、6 kernel sourcesを確認。
- statusは`KernelWorkerStatus.RUNNING`。以前のユーザー指示どおり継続pollingは行わず、完了連絡後に
  同じv1の通常logsを取得する。

## 2026-07-16 v4 GPU integrated train v1完了

### 実行確認

- kernel: `kentookumura/exp244-bidirectional-multiview-train` v1。
- Kaggle status: `KernelWorkerStatus.COMPLETE`。Traceback / Error / Exceptionなし。
- active variant 1 / LightGBM 3 configs / 5 folds / 15 boosters。
- parent/control再学習なし。validationはofficial-start 3,783,989 rowsのみ。
- runtime 17,201.149 sec、peak RSS 19,957.77 MB、features 380。
- official rows 3,783,989、pseudo 3,081 views / 770,157 rows、weight 0.5。

### OOF

| surface | raw exp218 | integrated | delta |
| --- | ---: | ---: | ---: |
| overall | 8.475794 | 8.472380 | -0.003414 |
| 000_050 | 0.957638 | 0.952745 | -0.004894 |
| 050_100 | 1.310177 | 1.304418 | -0.005759 |
| 100_250 | 2.094429 | 2.101476 | +0.007048 |
| 250_500 | 3.315459 | 3.358468 | +0.043009 |
| 500_1000 | 4.800747 | 4.863575 | +0.062828 |
| 1000_plus | 9.295198 | 9.286063 | -0.009135 |
| hidden-like spatial | 9.661607 | 9.245771 | -0.415836 |
| hidden-like typewell-purged | 9.636010 | 9.230900 | -0.405110 |

- fold delta: fold 0 `-0.470272`、fold 1 `+0.909638`、fold 2 `-0.033011`、
  fold 3 `+0.132699`、fold 4 `-0.601773`。改善3 / 5 folds。
- by-well: 387改善 / 386悪化。14 wellsが+2 ftを超えて悪化。
- worst `059c8f24`: 7.655552 -> 24.306119、delta `+16.650567`。
- 次点: `d90aa14c` +11.742932、`7987f2f2` +10.403484、`b37fd114` +9.339129。

### Adoption guard

- overall改善: PASS。
- 1000+非悪化: PASS。
- hidden-like spatial非悪化: PASS。
- hidden-like typewell-purged非悪化: PASS。
- 3 / 5 folds以上改善: PASS（ちょうど3）。
- worst-well +2 ft以内: FAIL（+16.650567）。
- `adoption_supported=false`。guardを緩和せず不採用とする。

### SHA / 取得物

- frozen exp218 OOF decompressed SHA: `5f3fc95182eea348f3545771e67778ce191e7ba468eee7b267f4993369422976`。
- feature schema SHA: `197c7ee8c296b9ef151931ce9127b0abe2856a6d5961f27ec24be29b7a9209b5`。
- training metrics SHA: `30422d567bcaf54fcac1697dcbfb21f709b25f6da8966c8cddb17f5d9c67e13d`。
- metrics SHA: `384640c0002486830c00a3caae67dd9c2ee40d95ce5ae63c9a6b798c48592f80`。
- by-well SHA: `5b31141ea4fb415bd8b1536eb5dc2e374511eb431a4cb39500bb2a398f01cede`。
- feature importance SHA: `02c4a48d6c57c319298e80543798590fa1296d98f4a1d76f77b0e2e52b3b469f`。
- model manifest SHA: `d93612c1c80d382f099892f08a34b4153b58554feb44cd17a69580256ccdb830`。
- prediction decompressed SHA: `3c4600562f385b80be1de7279d4bd52fb3de6f2e6db0570ba339d7b0e422e98b`。
- full output archiveは取得していない。原因監査に必要なmetrics / by-well / summaryだけを
  `kaggle/output/integrated_train_v1/`へselective downloadし、by-well / metrics SHA一致を確認した。
- inference prediction / submission: なし。

### 解釈と停止判断

380-feature schema、124 pseudo shards、outer-valid source-well除外、15 model、artifact SHAは契約どおりで、
実装エラーではない。mixed augmentationはhidden-likeと1000+を改善した一方、100-1000 rows、fold 1/3、
一部wellを大きく悪化させた。uniform pseudo weight 0.5がshifted-start適合とofficial-start局所分布の間で
不均一なtrade-offを作った可能性が高い。ただしearly/lateを同時投入したため方向別原因は未識別である。

mixed weightの事後grid、guard緩和、inference、submissionは行わない。再開する場合はexp244と同じcache・
sampling・weightでearly-onlyとlate-onlyを別々に学習するmatched attributionを先に行う。2 variants / 3 configs /
5 folds / 30 boosters相当になるため、別途明示承認が必要。
