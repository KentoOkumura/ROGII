# 設計

## 結論

exp497は、公開pipelineをexp413へ特徴追加する実験ではない。Public LB固有処理を
除いたpublic-coreを独立OOF trajectoryとして構築し、最後に保存済みexp413 OOFと
constant convex blendする。独立性と誤差相補性を確認してから、将来のexp413内蔵を
別仮説として判断する。

## 実験範囲

- 対象実験: exp497_strict_public_core_fold_safe_ensemble_on_exp413
- Route: ensemble
- 親実験: exp413_scale5_likpf_full_replacement_on_exp335
- 公開構造参照: raunakdey07/rogii-stacked-ensemble、id_no 128237178
- 参照source SHA256: 88c7b99e234fdbd5620c0045df294d9167eac84e56f538ceb3f2449a677a5454
- 既存source-port参照: exp082_public_artifact_replay_followup
- ensemble失敗比較: exp494_exp413_cat_xgb_physics_bounded_stack
- 変更する変数: 独立public-core OOFとexp413 OOFの予測レベルconstant blend
- 固定する変数: exp413 OOF/fold/score rows、RMSE、固定scope、tail guard、outer 5 folds

## strict public-core の構造

### 1. 物理候補bank

- likelihood-PF scale 3/5/8/12、Beam mean、last-known holdをtarget-freeに生成する。
- 公開sourceの6 selector variantとglobal variantを候補集合として固定する。
- 公開固定閾値 n_eval=4840、z_span=136.73/185.513、および公開bin mapは使わない。
- 各outer foldでouter-train wellsだけを使い、n_eval中央値とz_span 1/3・2/3分位で2×3 binを作る。
- 各binではouter-train pooled row SSEが最小の固定variantを選ぶ。tieは宣言順、support 40 wells未満はouter-train全体で選んだglobal winnerへfallbackする。
- outer-validのwell形状、物理候補、last-knownは利用できるが、outer-valid TVT/errorは選択完了まで読まない。

### 2. SP45 residual ML branch

- targetはTVT-last_known_tvt。
- 公開SP45 residual feature builderと3 LightGBM + 2 CatBoost + positive Ridge構造を再現する。
- 各outer foldのouter-trainをinner GroupKFold 4分割し、5 configs × 4 inner foldsを学習する。
- outer-train meta入力はinner OOF、outer-valid meta入力は4 inner model平均とする。
- positive Ridge alpha=1.66はouter-train inner OOFだけでfitする。
- 予定学習量は3 LGB + 2 Cat × outer 5 × inner 4 = 100 boosters、Ridge 5本。

### 3. SP45 anchorとU projection

- outer-train inner OOFだけから、residual MLとwell-shape physical selectorのconstant convex weightをfitし、outer-validへ適用する。
- 公開固定0.30/0.70はprimaryに使わず、ridge weightを[0.15, 0.45]へ制約する。
- U=TVT+Zをnormalized MD上のdegree 3 robust polynomialでfitする。4回の再重み付けを固定する。
- raw SP45とprojected pathのweightはouter-train inner OOFだけで[0.50, 1.00]から連続最小二乗fitし、outer-validへ適用する。公開固定0.75を直接使わない。

### 4. 独立learned trajectory branch

- public sourceのraw MD/Z/XY/derivative、GR rolling/diff/lag/lead、PF ANCC/Z、Beam、multi-scale NCC、LikPF scale、candidate disagreement、formation-plane KNN、dense ANCCを再構成する。
- FormationPlaneKNNとDenseANCCImputerのpoolからouter-valid wellsを除外する。
- SP45 residual branchとは別model manifestを持つ3 LightGBM + 2 CatBoost + positive Ridgeとする。
- outer 5 × inner 4 × 5 configs = 100 boosters、Ridge 5本。SP45 branchとの合計予定は200 boosters（LGB 120、Cat 80）、Ridge 10本。
- warmupは1-exp(-md_since/85)、Savitzky-Golayはwindow 61/poly 3を構造定数として固定する。
- model deltaとLikPF scale5 deltaのweightはouter-train inner OOFだけで[0.50, 0.80]からfitし、公開LB hedgeの0.60を直接使わない。

### 5. public-core内部blend

- projected SP45とlearned trajectoryのconstant convex weightを、各outer foldのouter-train inner OOFだけでfitする。
- projected SP45 weightの範囲は[0.50, 0.80]。公開profileの固定0.60は直接使わない。
- 生成されるouter-valid predictionをstrict_public_core_oofとする。

### 6. exp413とのfold-safe ensemble

- 保存済みexp413 final OOFをwell_id,row_idx,foldでstrict joinする。exp413を再学習しない。
- meta fold fでは、f以外のouter-fold OOF rowsだけでpublic-core weightを単一scalarとしてfitし、fold fへ適用する。
- 制約は0<=w_public_core<=0.30、w_exp413=1-w_public_core。intercept、row/well feature、gateは使わない。
- deployment weightは5 meta-fold weightのmedianとし、同じ[0,0.30]へclipする。full-OOFで再fitしない。
- primary scoreは5 meta-fold予測を結合したcross-fit blend。global same-OOF最適weightはoracle diagnosticとしても計算しない。

## Public LB特化として除外する処理

- well ID 00e12e8bその他の固定ID分岐
- Q0522、A27、+2.0 ft branch shift、total_shift_vs_6p809_ft
- public row count、public well count、public submission SHAを要求する分岐
- public outputやprecomputed submissionのcopy/fallback
- same-well train contact reconstructionとcontact override
- visible-prefix cut-frac候補評価、profile選択、final overlay
- bimodal branchのPublic-tuned trigger/hedgeとheel-based final override
- model-package tiny correction
- Public LB結果に基づくweight、threshold、variant、well選択

## 評価と固定gate

比較対象はexp413、public-core単体、cross-fit blendの3つ。全体、5 folds、distance
near/mid/far、0--250、1000+、raw GR observed/missing、high-missing、hidden-like
spatial/typewell-purged、by-wellを保存する。

primary AND gate:

1. pooled RMSE gain vs exp413 >= 0.03 ft
2. nonworse folds = 5/5
3. 全固定scope delta <= 0.00 ft
4. by-well delta p95 <= +0.25 ft
5. worst-well delta <= +0.25 ft
6. public-core weight > 0 in 5/5 meta folds、各weight <= 0.30
7. technical、leakage、row/fold/SHA contractが全PASS

FAIL時はexp413をselected predictionとして維持する。同じOOFでweight cap、内部weight、
selector threshold/map、smoothing、projection、feature/model subset、conditional gateを
救済しない。exp413への取り込み、inference、submissionへ進まない。

## 段階実行

- Stage 0: source/decontamination/row/fold/feature/PF seed contractの0-model preflight。
- Stage P: 2系統LikPF、PF ANCC/Z、Beam/NCC、well-shape候補を生成してtruth前にfreeze。
- Stage M1: SP45 residual nested models、100 boosters。
- Stage M2: independent learned nested models、100 boosters。
- Stage E: public-core内部weight、exp413 meta-fold blend、固定評価gate。新規booster 0。
- Stage I: gate PASS後の別承認時だけfull-train/current-test inferenceを同じexp497で扱う。

### 2026-08-03 Stage I override設計

ユーザーの明示overrideにより、gate FAIL後もexp497候補のprediction-only診断としてStage Iを
実行する。ただしtrain-side決定は再分類せず、selected anchorはexp413のままとする。

- train入力はSHA検証済みStage P outer0..4 union、3,783,989 rows / 773 wells。
- test base featuresはcurrent raw testからexp072 runtimeと同じstable seed policyで再生成する。
- exp497 selector/learned LikPFも`exp497::stage_p::*::<well>` seedでcurrent testへ再生成する。
- spatial imputerは全773 train wellsをpoolとし、test wellsだけをqueryする。Public row/well固定値を使わない。
- full-train modelはGroupKFold inner 4でOOF/testを同時生成する。2 branches ×
  (LGB 3 + Cat 2) × inner 4 = LGB 24 + Cat 16 = 40 boosters、Ridge 2。
- selectorはinner OOFでcross-fitし、test適用policyだけ全773 train wellsでfitする。
- 4内部weightはfull-train OOFだけでfitし、Stage Mと同じbounds、projection、warmup、smoothingを使う。
- exp413は既存current-test predictionを再利用し、再学習・再推論しない。
- final diagnostic blendはStage E meta5 weightのmedian `0.13716473330712417`を使う。
- 出力はcomponent prediction、feature/model/weight manifest、reproducibility summary。submission.csvと
  competition submitは生成しない。

### 2026-08-04 Stage I model serialization override

version 3は40 boostersをfitして予測後に破棄し、統計manifestだけを保存したため、hidden-test用の
inference-only deploymentへ分離できなかった。ユーザー承認により科学条件とfit inventoryを変えず、
version 4ではLightGBM 24本を`stage_i_models/*.txt`、CatBoost 16本を`*.cbm`、Ridge 2本を
`stage_i_ridge_weights.json`へ保存する。各boosterは保存直後に再読込してcurrent-test predictionの
最大絶対差`<=1e-5`を必須とし、相対path、SHA256、bytes、model-set SHAをmanifestへ記録する。
exp413再学習・再推論、submission生成は引き続き0。GPU rerun後もgate FAILとselected anchor
exp413は変更しない。

### 2026-08-04 saved-model hidden-safe inference設計

Stage I version 4 kernel outputをimmutable model sourceとし、manifest、Ridge、内部weight、selector
policy、feature schema、reproducibility manifestを個別SHAとmodel-set SHAでfail-closed検証する。
推論時のfit inventoryはexp497 booster 0、Ridge 0、exp413 booster 0。読み込む保存modelはexp497
40本とexp413 75本である。

raw hidden testからexp497 public replay / selector LikPF / learned LikPFをstable per-well seedで動的再生成し、
保存selector policy、40 booster、Ridge 2、固定内部weightを順に適用する。exp413も公開test固定予測を
使わず、exp510 version 4でhidden-safe修正済みのdynamic runtimeを再利用する。sample IDの集合・順序、
重複、finite、40 model SHA、feature schema、5 config × inner4 coverage、最終blend式を全AND監査する。
visible sampleとStage I v4参照予測のIDが一致する場合だけsaved-model parityを必須化し、hiddenでは
dynamic ID契約へ切り替える。`submission.csv`生成は許可するが外部submitは行わない。Stage E gate
FAILとselected train anchor exp413は変更しない。

### 2026-08-04 saved-model inference version 2 technical recovery

version 1は科学式・model推論ではなく、historical visible parityへstrict成分とdynamic exp413を含む
blendを同じ`0.001 ft`で判定して停止した。version 2はmodel、feature、weight、dynamic runtimeを変えず、
保存model再読込のstrict public-coreを`<=0.002 ft`、dynamic exp413を含むfinal blendを`<=0.020 ft`の
component別guardへ分離する。version 1実測`0.001281 / 0.014195 ft`は通すが、それぞれ独立に上限超過を
fail-closeする。exp413 historical差はreportに残す。exp413 helperが先に作るparent-only
`submission.csv`はID/order/finite/serialized exp413差`<=0.001 ft`を確認後、
`artifacts/exp413_intermediate_submission.csv`へ移動し、final exp497だけがworking直下の
`submission.csv`を生成する。fit、weight refit、外部submitはいずれも0。

2026-08-01にStage 0 compact preflightと専用contract testの実装だけが承認・完了した。
Stage P/M1/M2/E実装、正規Notebook採用、Kaggle package/runは未承認。Stage 0 inventoryで
確定したPF/Beam run数、feature数、memory見積り、200 boosters、親control再学習0を
SESSION_NOTESへ再掲し、次段へ進む前にユーザー承認を得る。

同日、その後のユーザー指示によりStage P/M1/M2/E実装、正規train Notebook採用、Kaggle
package/runも承認された。Stage PとStage EはKaggle CPU、Stage M outer0..4は各40 boostersを
Kaggle GPUで順次実行する。Colabは使用しない。inference/submissionは今回の範囲外とする。

## 再現性設計

- seed policy: SHA256(experiment, stage, split, outer_fold, inner_fold, family, well_id, seed_index)から局所seedを生成する。
- stochastic処理: likelihood-PF、PF ANCC/Z、LightGBM、CatBoost。
- PF/Beam: PFは500 particles × 128 seedsを基本契約とし、global RNGを禁止する。Beam/NCCは決定的順序を固定する。
- 並列処理: well単位seedを事前生成し、thread/joblib schedulingで乱数系列を変えない。
- GPU: deterministic flags、固定thread、device/dtypeをconfigに明示する。bitwise deterministicとは主張せず、model/prediction SHAを記録する。
- SHA: raw/decompressed input、source、feature schema/content、candidate、outer/inner fold、model manifest、各component OOF、blend weight、final OOFを記録する。
- inferenceへ進む場合だけtest feature/prediction/submission SHAとfallback rowsを追加する。
- Kaggle package: push前にmetadataとbootstrap内config、source SHA、selected stage、model countを照合する。
- deterministic anchor: 独立rerun一致までfalse。

## リスク

- リークリスク: spatial imputer pool、well-shape map、内部weight、Ridge metaがouter-valid truthへ触れる危険。全てouter-train inner OOF内へ限定する。
- CV/LB不一致: exp494はCV改善5/5でもLB悪化した。foldだけでなくscope/tailをAND gateにする。
- ランタイム: 予定200 GPU boostersに加え2系統128-seed PFが重い。Stage分割と別承認を必須にする。
- メモリ: 378万行の2 feature surfaceを同時保持しない。fold/partition単位保存を設計する。
- 再現性: public sourceのNumba/global RNGをそのまま移植せずstable per-well seedへ変更するため、public outputとのbyte parityは目的にしない。
- 独立性: 生データと物理primitiveの一部はexp413と共通し得るが、final370、selector、最終モデル、予測を共有しない。相関は結果として測る。

## 次のアクション

Stage PをKaggle CPUで実行してSHAを確認し、Stage M outer0..4をKaggle GPUで順次実行する。
5 shard成功後にStage Eの固定gateを判定する。
