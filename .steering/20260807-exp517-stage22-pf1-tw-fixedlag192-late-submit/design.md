# 設計

## 2026-08-08 v2修正設計（現行）

v1は`pf_1`単体meanをsubmissionへ直接decodeしたため、writeupのstage 2-2である`5-input + fixed-lag smoother + tabular model`を実装していなかった。この不一致を名称変更で処理せず、同一exp内の失敗履歴として固定したうえで実装を修正する。

修正版は公開v96 configのoriginal Optuna group先頭5 bankを採用する。`pf_1 / pf_2 / pf_3 / r0_seed32 / r1_seed32`を、bankごとの公開particle数と32 seedsで`twGR`へ適用する。全bankでfixed-lag 192を使い、stage 2-2より後のStudent-t、tempering、ps-combo、anchor、learned emission、self/nbr、whole smoothingは無効化する。

tabular側は公開Ravaghi artifactのbase feature schemaを再利用し、baseのPF aliasを`pf_1` smootherへ置換する。5 trajectoriesは、公開baseが1 PFへ作るabsolute mean、std、last-known delta、spatial/dense差、typewell-GR offset residualの同型featureとしてbank suffix付きで連結する。targetは`TVT-last_known_tvt`。公開の3 LightGBM / 2 CatBoost / positive Ridge / `alpha=1,tau=85,w_pf=0.09` / SG(17,3) decodeを維持する。

学習するのは修正variant 1本だけで、3 LightGBM + 2 CatBoost × 5 folds = 25 base models、Ridge 5 folds。公開artifact control、exp516、exp517 v1は再学習しない。CV `7.50`近傍のscore再現をgateとし、不一致なら同じ契約内の実装差を監査する。LBに合わせたparameter/weight調整は行わない。

## アプローチ

exp516のcompact self-contained inferenceを構造参照元とし、公開Notebookから抽出済みのPF engineとdynamic hidden-test guardだけを再利用する。anchor生成とlearned-emission生成を実行経路から外し、vendor configの`pf_1`へ`smooth_mode=fixedlag`、`smooth_lag=192`を明示して`twGR`だけを実行する。PF meanを追加融合せず直接submissionへ整列する。

親exp516のlate submissionは`pfA + anchor + emission + full smoothing`という別契約なのでcontrolを再実行しない。比較値は保存済みexp516 Public `10.056` / Private `8.552`を参考表示するだけで、stage 2-2のpublished system scoreとは直接比較しない。

## 手法忠実性

- 実装区分: `proxy`（ユーザー承認済み）。
- 一次資料との一致点: GPU bootstrap PF、twGR observation、32 seeds、fixed-lag ancestral smoothing、lag 192、single-well suffix context。
- 参照sourceとの一致点: 公開PF engine、state `TVT+Z`、GR power likelihood、resampling/jump/transition、seed likelihood soft aggregation、最終公開`pf_1`の600-particle parameter set。
- 参照sourceからの変更 / 省略点: 最終v96 configのglobal `smooth_mode=full`をstage 2-2記載の`fixedlag`へ変更する。final-v96のparameterをstage 2-2当時の非公開parameterのproxyとして使う。5-input tabular model、残り4 PF config、anchor、emission、self/nbr、full smoothingを省略する。
- input tensor / feature: hidden suffixの`MD, Z, GR`、typewell TVT/GR grid、known-prefix終端position/rate。
- target / objective: state position/rate posterior approximation。学習objectiveなし。
- output representation: per-row fixed-lag smoothed TVT mean/stdとrun log-likelihood。
- loss: 学習lossなし。GR observation likelihoodのみ。
- decode / postprocess: 600 particles × 32 seeds、lag 192 ancestry backtrace、seed log-likelihood soft average。gain/blend/projectionなし。
- context unitと予測範囲: hidden suffix内のforward historyと最大192 future rows。whole-interval futureは使わない。
- 支持 / 棄却できる主張: 最終公開`pf_1` parameterをstage 2-2型fixed-lag-192/twGR/single-outputとしてhidden code submission上で再生成したlate score。
- 判断できない主張: stage 2-2 published 5-input+tabular score、当時のexact `pf_1`、5-inputのsmoother gain、6位最終system。
- 実験名との整合: `stage22_pf1_tw_fixedlag192_late_submit`は目標stage、proxy bank、representation、smoother、late phaseだけを表す。

## 探索幅とpivot判定

- 変更class: `mechanism`。exp516のphysical-anchor/full-smoother componentから、stage 2-2型GR-only fixed-lag componentへ契約を変更する。
- 同じ親 / familyの連続小改善: exp516からのparameter tuningではなく、ユーザー指定でhistorical stageそのものを変更する1実験目。LB救済gridは行わない。
- positive evidence: writeupは5-input systemについてlag 192がCV `-0.3`、LB `-0.4`と報告し、`pf_1 × twGR`のfilter/smoother例を示す。
- representation案比較: full final 91 candidates、stage 2-4 prior PF、stage 2-2 fixed-lag PF。ユーザーがstage 2-2 PF単体proxyを選択した。
- pivot根拠: exp516 mismatchのparameter rescueではなく、比較対象とsource eraをstage 2-2へ切り替える。
- `kaggle-idea-forge`: 不要。新規案探索ではなく、ユーザー指定手法の固定proxy replayである。

## 実験範囲

- 対象実験: `exp517_stage22_pf1_tw_fixedlag192_late_submit`
- Route: `pf_beam`
- 親実験: 実装parent `exp516_sixth_place_pfa_tw_late_submit`、科学sourceはdiscussion stage 2-2と公開final Notebook。
- 変更する変数: bank `pfA -> pf_1`、anchor `ON -> OFF`、emission `ON -> OFF`、smoothing `full -> fixedlag 192`。
- 固定する変数: twGR、600 particles、32 seeds、PF seed `4423098`、dynamic sample alignment、T4 x2、internet off、one-shot late submission。

## 再現性設計

- seed policy: 公開source/exp516と同じPF seed `4423098`とsorted well orderを固定し、chunk/device数をmanifestへ保存する。
- stochastic処理: PF transition/jump/resamplingにあり。anchor/ML trainingはない。
- PF/Beam / seed bagging: PFあり、Beamなし。32 seedsをrun log-likelihoodでsoft aggregationする。
- 並列と乱数: 公開GPU PFはchunkごとgeneratorを初期化しdevice/chunk分配の影響を受けうる。T4 x2、`PF_NGPU=2`、well sort、chunk policyを固定するがbitwise determinismは未保証。
- runtime: Kaggle T4 x2、float32、internet off。fixed-lag history bufferはfull ancestryより軽いが、hidden well長に応じて可変。
- test regeneration SHA: source/config SHA、well/row count、prediction logical content SHA、submission SHAをexecution manifestへ保存する。
- model manifest: modelなし。PF config、source、runtime、candidate、prediction、submissionのmanifestを保存する。
- Kaggle bootstrap: canonical notebookとsupport configをprepare後に再生成し、metadata T4/internet off、slug/title、bootstrap config、source SHAをpush前に照合する。

## リスク

- leakage: PFはruntime test suffix GR/typewell/known prefixだけを使い、train target/anchor/neighborを使わない。sample IDを提出schemaの正とする。
- CV/LB不一致: published 6.724/7.404は5 PF+tabularであり、本proxyとのscore比較は禁止する。
- runtime/memory: fixed-lag192はlag16より約2.9倍とwriteupで報告。visible testは小さいがhiddenの可変well数・長さを前提にchunk処理する。
- 再現性: GPU RNG/reductionとchunk境界で揺れうる。単発結果をdeterministic anchorと呼ばない。
- 手法忠実性: stage 2-2 exact config/5 PF/tabularが非公開。final `pf_1`への置換を常にproxyと表示する。
- 過度なproxy化: `pf_1/tw/600/32/fixedlag192/direct mean`を固定し、LB後調整しない。これ以上の省略・代替が必要なら再承認を得る。
