---
title: 関連研究調査 ROGII Wellbore Geology Prediction
date: 2026-05-28
types:
  - survey
  - literature_review
experiments: []
topics:
  - well_log_correlation
  - geosteering
  - gr_matching
  - formation_surface
  - sequence_model
status: final
summary: "well-log相関、geosteering、formation推定などの関連研究を整理し、ROGIIへ転用可能な候補と検証上の注意をまとめた。"
---

# 関連研究調査: ROGII Wellbore Geology Prediction

- 対応する上位仮説: なし

調査日: 2026-05-28

## クエリ

- コンペ/タスク: ROGII - Wellbore Geology Prediction。horizontal well の evaluation zone における `TVT` 回帰、RMSE minimize。
- 検索語:
  - `well log correlation gamma ray dynamic time warping geosteering paper`
  - `automated geosteering machine learning gamma ray logs neural network paper`
  - `formation tops prediction machine learning well logs gamma ray paper`
  - `FORCE 2020 lithology prediction competition winning solution well logs`
  - `Deep Hierarchical Graph Correlation well-log alignment CNN dynamic programming`
- 比較したローカル情報:
  - `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending/solution_summary.md`
  - `docs/notebooks/rogii-wellbore-geology-prediction/eda_insights_summary.md`
  - `docs/discussions/rogii_discussion_insights_20260528.md`

## 実行価値の高いアイデア

| 優先度 | アイデア | 出典 | 期待効果 | 難度 | リスク |
|---:|---|---|---|---|---|
| 1 | CWT / multi-scale GR texture を NCC / deterministic DTW の add-only 特徴にする | Behdad 2019, AeroRidge, Mitch | GR の amplitude/noise に強い alignment signal を増やす。既存 PF/beam/NCC pipeline を壊しにくい。 | 中 | DTW は計算量が重い。TVT が reverse / flat / fold-back する well では単純な monotonic path が外れる。 |
| 2 | fold-safe formation top recommender / spatial surface imputer | Pisel 2022, Konbu17, Mitch | train-only formation columns を直接使わず、`(X,Y,Z,trajectory,prefix)` から ANCC 等の構造面を推定する。 | 中 | validation well を含めて fit すると leakage。hidden test が近傍不足なら悪化。 |
| 3 | learned GR-window similarity + constrained DP を feature generator として試す | Acharya et al. 2026 DHGC | raw NCC/DTW より noisy GR・振幅差に強い alignment cost を作れる可能性。 | 高 | GPU/学習コストが大きい。ROGII の direction 反転や non-monotone TVT に制約を合わせる必要がある。 |
| 4 | top-k trajectory / uncertainty feature を作る | Alyaev & Elsheikh 2022, Kaggle discussion | 単一予測が外れる hard well に対して、候補パスの分散・順位・agreement を tree model と postprocess に渡せる。 | 高 | RMSE では平均化が有利な場面も多い。candidate path の OOF が弱いとノイズ特徴になる。 |
| 5 | FORCE 2020 由来の blind-shift 対策を運用ルールに入れる | FORCE 2020 results, Equinor repo | regional cluster、model disagreement、well-level blind validation の考え方を ROGII CV/LB 判断に転用する。 | 低 | FORCE は lithology classification であり、ROGII の TVT regression とは metric/target が違う。 |

## まず採用するなら

最初に実験へ落とす価値が高いのは、`CWT / deterministic alignment features` と `formation surface imputer` の 2 つ。どちらも Kaggle 公開ノートブックで既に効いている signal family と整合し、外部研究の追加分を add-only にしやすい。

深層モデル系は「本命実装」ではなく、まず feature generator として使う。現時点で ROGII の public notebooks は tree ensemble + drift target + NCC / formation / PF / beam が強く、sequence model を主役にする根拠は弱い。

## 2026-06-04 追補: 最新 public methods と関連研究の対応

2026-06-03 の public notebook / discussion 更新後に、実際に上位化している手法 family と論文・既存研究の対応を追加確認した。結論は、既存の 2026-05-28 調査は大枠では有効だが、最新上位の `PF / physical sel15`、`PF / beam / TabICL stack`、`AeroRidge v34` 系に対しては、particle filter / ensemble uncertainty / tabular foundation model の論文対応を明示する必要があった。

### 追加で実行価値が高いアイデア

| 優先度 | アイデア | 関連する public method | 関連研究 | 期待効果 | リスク |
|---:|---|---|---|---|---|
| 1 | PF / beam の候補分布を OOF feature 化する | Aiden / Needless / Safar sel15、PF scale selector、beam / hold blend | Muhammad et al. 2024, Jahani et al. 2022, Alyaev & Elsheikh 2022 | 単一 PF submission を追うより、scale 別予測、likelihood、候補間分散、cost margin を tree/postprocess に渡せる | PF 128/256 seeds は hidden 200 wells で重い。seed 数・particle 数・scale を config 管理しないと再現性が落ちる |
| 2 | DTW / DWT / wavelet alignment は PF/beam の補助 signal として使う | AeroRidge v34、DWT-DTW、Mitch NCC | Behdad 2019, WFT-DTW 2025, DHGC 2026 | GR の noisy / amplitude shift に対して、banded DTW cost、best lag、path slope、scale energy を add-only で足せる | full DTW は O(NM) で重い。`exp008_gr_ncc_matcher` は悪化済みなので、GR alignment 単独ではなく PF/beam 候補の診断に限定する |
| 3 | formation / physical branch は hidden-safe imputer に落とす | Sunny physical、見えている train well 用の物理処理、Konbu17 plane-fit | Pisel et al. 2022, Elahifar & Hosseini 2024 | train-only formation columns を直接使わず、fold-safe spatial/top recommender と uncertainty に変換できる | hidden test では formation columns / Geology を期待しない。validation well を含めた fit は leakage |
| 4 | TabICL / 生成物を使った積み上げ は主役ではなく diversity source として扱う | PF/beam/TabICL stack、v10 生成物を使った積み上げ | TabICL 2025, LightGBM 2017, CatBoost 2018 | LightGBM/CatBoost と異なる inductive bias の候補として blend / pairwise distance 診断に使える | TabICL は GPU・artifact・version 依存が強い。ROGII は regression task なので classification benchmark の期待値を過大評価しない |
| 5 | SG smoothing / hold blend / fade-in は signal processing postprocess として OOF 監査する | SG smoothing、hold-last-known blend、alpha/tau fade-in | Savitzky & Golay 1964 | prediction-start continuity と局所ノイズ抑制に効く可能性 | OOF-fit の自由 alpha / tau / row bucket は楽観的になりやすい。nested CV か固定候補だけで比較する |

### 追加情報源

#### High-Precision Geosteering via Reinforcement Learning and Particle Filters

- URL: https://arxiv.org/abs/2402.06377
- 日付/年: 2024
- 対応する public method: Aiden / Needless / Safar 系の PF ensemble、scale selector、likelihood-weighted path。
- 事実: geosteering で particle filter を state estimation に使い、real-time well-log data から地層境界に対する位置を推定する構成。
- 転用仮説: ROGII では RL 部分を使わず、PF の候補パス、likelihood、scale 別 mean/std/range、beam との差分を OOF feature / diagnostic にする。
- 実装メモ: `n_particles`, `n_seeds`, `scale`, `resampling`, `GR interpolation`, `hold blend` を config 化する。
- leakage/実行時間/オフライン実行リスク: hidden 約 200 wells で runtime が支配的。public visible 3 wells の所要時間は信用しない。

#### Ensemble-based well-log interpretation and uncertainty quantification for well geosteering

- URLs:
  - https://doi.org/10.1190/geo2021-0151.1
  - https://norceresearch.brage.unit.no/norceresearch-xmlui/handle/11250/2998569
- 日付/年: 2022
- 対応する public method: PF / beam / candidate ensemble / uncertainty feature。
- 事実: ensemble randomized maximum likelihood 系の iterative inversion で layer boundary と petrophysical properties の不確実性を扱う。
- 転用仮説: ROGII では物理 forward simulator を作るより、PF/beam/DTW/formation estimator の disagreement を uncertainty proxy にする。
- 実装メモ: candidate mean だけでなく、std、range、top1-top2 margin、rank entropy、known-prefix fit error を保存する。
- leakage/実行時間/オフライン実行リスク: uncertainty feature は OOF snapshot が fold-safe でなければ過信しやすい。

#### WFT-DTW stratigraphic correlation / semi-automatic DTW well-log correlation

- URLs:
  - https://www.sciencedirect.com/science/article/pii/S2666759225000332
  - https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/be83ab3ecd0db773eb2dc1b0a17836a1Expanded_Abstract_Nilo_SBGF.pdf
- 日付/年: 2025 / 2023
- 対応する public method: AeroRidge DWT-DTW、Mitch NCC、deterministic DTW alignment。
- 事実: wavelet frequency decomposition と DTW、または Sakoe-Chiba band 付き DTW を well-log / stratigraphic correlation に使う研究がある。
- 転用仮説: ROGII では full auto-correlation を狙わず、typewell GR に対する scale 別 cost、best lag、path slope、alignment confidence を特徴にする。
- 実装メモ: deterministic banded DTW を優先し、stochastic feature snapshot mismatch を避ける。
- leakage/実行時間/オフライン実行リスク: validation tail の true TVT を alignment target にしない。GR NaN interpolation の前後で cost が壊れていないか見る。

#### TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

- URL: https://arxiv.org/abs/2502.05564
- 日付/年: 2025
- 対応する public method: PF/beam/TabICL stack、生成物を使った積み上げ。
- 事実: TabICL は大きめの tabular data に対する in-context learning を狙う tabular foundation model。
- 転用仮説: ROGII では TabICL を単体本命にせず、LightGBM / CatBoost / PF features と誤差相関が低いかを artifact diversity として確認する。
- 実装メモ: candidate output の pairwise distance、per-well range、public score だけでなく OOF 相関を記録する。
- leakage/実行時間/オフライン実行リスク: external artifact / pretrained model 依存は Kaggle offline input と version を固定する。classification benchmark の強さを TVT regression に直結させない。

#### Savitzky-Golay smoothing and GBDT baselines

- URLs:
  - https://pubs.acs.org/doi/10.1021/ac60214a047
  - https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision
  - https://papers.neurips.cc/paper/7898-catboost-unbiased-
- 日付/年: 1964 / 2017 / 2018
- 対応する public method: SG smoothing、LightGBM/CatBoost residual model、hill climbing blend。
- 事実: SG smoothing は局所多項式 least-squares smoothing、LightGBM/CatBoost は tabular GBDT の標準的な強い実装。
- 転用仮説: ROGII の tree ensemble は引き続き主力。SG は予測曲線の局所ノイズを落とすが、地質的変化を消す可能性もあるため OOF 監査が必要。
- 実装メモ: smoothing window、polyorder、prediction-start fade-in、hold-last-known weight を固定候補として比較する。
- leakage/実行時間/オフライン実行リスク: OOF 全体で最適化した smoothing / alpha は楽観的。public 3 wells に合わせた window は採用しない。

## 情報源

### ROGII official competition

- URL: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- 日付/年: 2026
- タスク/metric: horizontal well evaluation zone の `tvt` 回帰。公式ページ本文では RMSE。
- 手法: 公式情報。Notebook-only code competition、Kaggle inference は internet disabled。
- 主な結果: 公開 `test/` は実行確認用の少数 well。提出時は hidden test に差し替え。
- 転用仮説: external data / pretrained models は許可されるが、offline notebook で再現できる形にする必要がある。
- 実装メモ: `submission.csv` は `id,tvt`。CV は GroupKFold by well、`TVT_input.isna()` rows のみ score。
- leakage/実行時間/オフライン実行リスク: public visible test への coordinate overlap や static CSV 依存は hidden-safe ではない。

### Kaggle public notebooks: target-free / drift / NCC / formation / DTW

- URLs:
  - https://www.kaggle.com/code/pilkwang/rogii-eda-target-free-alignment-for-tvt
  - https://www.kaggle.com/code/mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore
  - https://www.kaggle.com/code/mitchgansemer/gr-features-outlier-detection-rogii-wellbore
  - https://www.kaggle.com/code/konbu17/rogii-plane-fit-formation-top-knn
  - https://www.kaggle.com/code/svanikkolli/aeroridge-engine-v2
  - https://www.kaggle.com/code/ravaghi/wellbore-geology-prediction-hill-climbing
- 日付/年: 2026。ローカル取得: 2026-05-27 / 2026-05-28。
- タスク/metric: ROGII `tvt` RMSE。
- 手法: drift target、multi-scale NCC、formation KNN / plane fit、PF / beam、DWT / deterministic DTW、GBDT / HGB / CatBoost / XGBoost、OOF blend、conservative postprocess。
- 主な結果:
  - Mitch 系は `TVT - last_anchor_tvt`、GR NCC、formation KNN、PF/beam、HGB blend の改善因果が最も明確。
  - AeroRidge は stochastic feature snapshot を避け、deterministic DTW を add-only で足す方針が実験設計として参考になる。
  - Pilkwang は strict drilling-time と offline batch の情報境界を整理している。
- 転用仮説: 外部論文は、既存の強い解法ファミリーを置き換えるより、alignment / formation / uncertainty features の設計を補強する用途がよい。
- 実装メモ: raw TVT ではなく drift / residual target を標準にする。formation columns は direct feature にせず fold-safe imputer 経由。
- leakage/実行時間/オフライン実行リスク: public-only blend、static public CSV、validation well を含む imputer fit、train-only columns 直利用に注意。

### A step toward practical stratigraphic automatic correlation of well logs using CWT and DTW

- URL: https://www.sciencedirect.com/science/article/pii/S0926985118304336
- 日付/年: 2019
- タスク/metric: stratigraphic well-log correlation。GR log の CWT 由来 spectral trend と DTW による correlation。
- 手法: Continuous Wavelet Transform、polynomial normalization、DTW、GR の normalized power image による確認。
- 主な結果: CWT で作った trend logs を DTW に入力し、自動 well-log trend correlation に使えることを示している。
- 事実: GR は facies-sensitive な time series として扱われ、CWT は local texture / trend を抽出する。
- 転用仮説: ROGII では `GR` の rolling stats だけでなく、CWT scale energy、CWT trend log、CWT-DTW best lag/cost/confidence を typewell alignment と lateral self-alignment の feature にできる。
- 実装メモ: まず `pywt` が使えるか確認。使えない場合は scipy convolution / Ricker wavelet / DWT で代替。DTW は full path ではなく banded deterministic DTW に制限する。
- leakage/実行時間/オフライン実行リスク: full DTW は O(NM) で重い。Kaggle offline で依存を増やしすぎない。validation fold ごとに feature generation を固定し、hidden tail true TVT は絶対に使わない。

### Deep Hierarchical Graph Correlation: CNN + Dynamic Programming for well-log alignment

- URL: https://www.mdpi.com/2077-1312/14/1/66
- 日付/年: 2026
- タスク/metric: well-log depth alignment。Pearson correlation、Euclidean distance、Earth Mover's Distance、成功率、DTW 比 runtime。
- 手法: 1D CNN triplet embedding、pair scoring network、constrained graph pathfinding / dynamic programming。
- 主な結果: unseen GR log pairs で高い alignment 成功率を報告。DTW より速く、unconstrained DTW の one-to-many mapping を抑える設計。
- 事実: raw sample distance ではなく learned local similarity matrix を作り、その後に geologically plausible path constraint をかける構成。
- 転用仮説: ROGII train では horizontal `GR` と typewell `GR` を true `TVT` 軸で対応付けられるため、positive/negative window pairs を作って learned similarity feature を作れる。
- 実装メモ: 本格モデルの前に、128/256 sample window の SimCLR/triplet-lite embeddingを作り、OOF で `best_match_tvt`, `path_cost`, `path_slope`, `cost_margin` だけ tree features に渡す。
- leakage/実行時間/オフライン実行リスク: positive pair 作成で validation well の true tail TVT を使うと leakage。fold ごとに embedding train を切るか、pretrain fold と validation fold を明確に分ける。TVT が non-monotone な区間は path constraints を緩めるか segment 単位にする。

### A recommender system for automatic picking of subsurface formation tops

- URL: https://arxiv.org/abs/2202.08869
- 日付/年: 2022
- タスク/metric: formation top picking。MAE / RMSE、four-fold CV。
- 手法: geophysical logs を使わず、既に pick 済みの formation tops から未 pick tops を推薦する recommender。
- 主な結果: spline interpolation と競合または上回る結果を報告。training data が増えるほど error variance が下がる。
- 事実: formation top は、ログ波形がなくても well 間の既知 top table から spatial / relational に推定できる。
- 転用仮説: ROGII の train-only `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` を、direct feature ではなく fold-safe recommender / imputer の teacher として使う。
- 実装メモ: per-fold で train wells だけから `(X,Y,well trajectory,typewell id/coordinates)` -> formation top を KNN / local plane / matrix factorization で推定。prefix `TVT_input` から well-specific bias を合わせる。
- leakage/実行時間/オフライン実行リスク: 全 train wells で imputer を fit して validation fold に使うと leakage。hidden test の地域が train convex hull 外なら uncertainty を大きくし、model に distance/confidence を渡す。

### Automated real-time prediction of geological formation tops on the Norwegian Continental Shelf

- URL: https://link.springer.com/article/10.1007/s13202-024-01789-5
- 日付/年: 2024
- タスク/metric: formation top classification。accuracy / precision / recall / F1、blind dataset。
- 手法: SVM、KNN、Random Forest、MLP、DBSCAN baseline、SHAP / permutation importance。
- 主な結果: MLP が blind dataset で高 accuracy。feature importance では depth, RPM, hook-load が強いと報告。
- 事実: blind dataset を別に持ち、DBSCAN のような spatial clustering baseline と比較している点が実運用寄り。
- 転用仮説: ROGII では RPM/HKLD はないため、手法そのものより「blind holdout」「spatial baseline」「feature importanceで geology signal を確認する」運用を借りる。
- 実装メモ: GroupKFold に加え、coordinate cluster holdout を 1 つ用意し、formation imputer と NCC features の robustness を確認する。
- leakage/実行時間/オフライン実行リスク: published accuracy は classification task で、ROGII の continuous TVT regression には過大評価しない。

### Machine Learning-Based Real-Time Prediction of Formation Lithology and Tops with GeoVision

- URL: https://www.mdpi.com/2673-4117/4/3/139
- 日付/年: 2023
- タスク/metric: lithology / formation top classification。accuracy、precision、F1。
- 手法: drilling parameters から lithology / top を予測。GeoVision web app。
- 主な結果: Volve field dataset で高い test accuracy を報告。
- 事実: ROP, WOB, RPM, mud/gas/flow 系など、drilling parameters の組み合わせが lithology/top prediction に効く。
- 転用仮説: ROGII にはこれらの surface drilling parameters がないため、直接転用しない。`MD`, `X`, `Y`, `Z`, `GR`, prefix `TVT_input` だけで完結する feature set に集中する判断材料にする。
- 実装メモ: 外部データを足すなら、publicly available pre-trained well-log model より先に、コンペ内 signals の fold-safe 実装を固める。
- leakage/実行時間/オフライン実行リスク: 外部 drilling dataset を使っても feature mismatch が大きい。Kaggle offline で web app 依存は不可。

### Applications of classification ML to predict formation tops and lithology while drilling

- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10652382/
- 日付/年: 2023
- タスク/metric: two-well Middle East dataset で lithology / formation top classification。accuracy、precision、recall、F1。
- 手法: Gaussian Naive Bayes、Logistic Regression、Linear Discriminant Analysis。ROP, DSR, Q, SPP, WOB, torque など。
- 主な結果: unseen well validation で GNB / LR / LDA が高 accuracy を報告。
- 事実: unseen well で validation している点は ROGII の well-level CV と近い。
- 転用仮説: low-capacity model を baseline として使い、complex model の改善が geology signal 由来かを確認する運用は使える。
- 実装メモ: formation classification ではなく、ROGII では `last_anchor_tvt` drift の simple ridge / HGB / CatBoost baseline と比較する。
- leakage/実行時間/オフライン実行リスク: drilling mechanics features が ROGII にない。classification accuracy を RMSE 改善に直結させない。

### Direct multi-modal inversion of geophysical logs using deep learning

- URL: https://arxiv.org/abs/2201.01871
- 日付/年: 2022
- タスク/metric: gamma-ray logs の real-time stratigraphic inversion。multi-modal probabilistic prediction。
- 手法: mixture density DNN、multiple-trajectory-prediction loss、複数の likely stratigraphic curves と確率を出力。
- 主な結果: deterministic regression より realistic な multi-modal prediction を出せる、という proof-of-concept。
- 事実: geosteering では単一解でなく複数候補と確率を扱う発想が自然。
- 転用仮説: ROGII の hard well では、NCC/formation/PF/beam/DTW が互いに食い違う。この disagreement を candidate path ensemble として特徴化すると RMSE の tail を削れる可能性がある。
- 実装メモ: 深層 MDN をすぐ作らず、まず既存 estimator の top-k path / cost margin / variance / entropy を tree features に入れる。
- leakage/実行時間/オフライン実行リスク: deep model は synthetic geology への過適合や seed variance が大きい。Kaggle runtime と artifact サイズに注意。

### FORCE 2020 lithology competition results

- URL: https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/
- 日付/年: 2020 results page, updated 2022-11-19
- タスク/metric: well-log lithofacies classification。blind well dataset。
- 手法: public competition summary。トップ解法と blind shift の分析。
- 主な結果: 329 teams、148 submitted、2200 scored submissions。blind dataset では train/test と分布が違い、過学習と分布差が final score に効いたと説明されている。
- 事実: 実 well-log task では label/log uncertainty と blind distribution shift が大きく、上位モデル間の差も小さい。
- 転用仮説: ROGII でも Public LB の絶対値より、GroupKFold / coordinate-cluster holdout / hard-well breakdown の一貫性を優先する。
- 実装メモ: fold ごとに per-well RMSE distribution、GR missingness、tail length、coordinate cluster、estimator disagreement 別に error を見る。
- leakage/実行時間/オフライン実行リスク: FORCE は classification で ROGII は regression。直接の architecture 転用ではなく validation discipline の転用。

### FORCE 2020 dataset and Equinor starter repository

- URLs:
  - https://zenodo.org/records/4351156
  - https://github.com/equinor/force-ml-2020-wells
- 日付/年: dataset published 2020-12-18。repo archived 2025-03-27。
- タスク/metric: well logs から lithofacies labels を予測。
- 手法: 118 wells dataset。Equinor repo は EDA、preprocessing、feature engineering、XGB/CatBoost、regional clustering を含む。
- 主な結果: repo README は XGB / CatBoost と regional clusters で local geology を捉える戦略を説明。
- 事実: well-level location and lithostratigraphy を使う classic GBDT pipeline は、今の ROGII public notebooks と近い。
- 転用仮説: ROGII の `(X,Y)` と typewell / trajectory から regional cluster を作り、fold error と feature importance を cluster 別に見る。
- 実装メモ: clustering は prediction feature として入れる前に、validation diagnostic として使う。feature にする場合は fold-safe に fit。
- leakage/実行時間/オフライン実行リスク: external FORCE data は basin/label が違うため pretraining 価値は低い。使うなら workflow reference に留める。

## 実験候補

1. 仮説: CWT / deterministic DTW は noisy GR の alignment feature として NCC を補完する。
   変更: existing drift baseline に CWT scale energy、banded DTW cost、best lag、cost margin、path slope を add-only 追加する。
   検証: GroupKFold by well、evaluation zone only。まず 1 fold / small wells で runtime と alignment の sanity plot を確認。
   停止基準: OOF 改善が 0.1 ft 未満、または per-well hard tail が増える場合は破棄。

2. 仮説: formation tops は direct columns ではなく spatial recommender として使えば hidden-safe に効く。
   変更: fold-safe KNN / local plane / low-rank imputer で `ANCC` などを推定し、`-Z + formation_hat + prefix_bias` と uncertainty を features に追加する。
   検証: fold 内で validation wells を完全除外して imputer fit。coordinate cluster 別 RMSE と distance-to-neighbor 別 RMSE を見る。
   停止基準: CV は改善しても、nearest-neighbor distance が大きい well で悪化する場合は confidence gating を入れるまで採用しない。

3. 仮説: learned GR-window similarity は raw NCC/DTW の ambiguity を減らす。
   変更: train folds の true TVT alignment から positive/negative GR windows を作り、小さな 1D CNN embedding を学習。DP path ではなく similarity summary を GBDT features にする。
   検証: 1 fold only で leakage audit。embedding train に validation well を入れない。NCC-only と同じ feature slots で比較。
   停止基準: OOF 改善が CWT/DTW add-only より小さい、または runtime が Kaggle 9h budget の 25% を超える。

4. 仮説: candidate path disagreement は hard well の RMSE tail を削る。
   変更: NCC、formation surface、PF、beam、DTW の top-k path / prediction を保存し、mean/std/range/rank/cost entropy を features にする。
   検証: per-well RMSE 上位 5% で改善を見る。全体 RMSE だけでなく tail RMSE と median RMSE を記録。
   停止基準: median は改善して tail が悪化する場合、postprocess ではなく feature gating を見直す。

5. 仮説: FORCE 2020 型の blind-shift discipline は ROGII の Public LB 過信を減らす。
   変更: GroupKFold に加え、coordinate-cluster holdout と hard-well report を `result.md` テンプレに入れる。
   検証: 既存 exp001/exp002 の metrics に cluster/hard-well breakdown を追加。
   停止基準: 実験速度が落ちすぎる場合は full report を提出候補のみに限定する。

## 調査上の判断

- Papers With Code で ROGII に直接対応する benchmark は見つからなかったため、公式 benchmark 相当として FORCE 2020 を参照した。
- 外部論文の多くは formation top / lithology classification であり、ROGII の continuous `TVT` RMSE とは直接対応しない。したがって論文の reported accuracy をスコア期待値として扱わない。
- Kaggle 公開ノートブックが示す制約に反する手法、特に public-only CSV blend、visible test overlap、train-only formation columns 直利用は実験候補から外す。
- 今の優先順位は、強い public solution family と整合する target-free features を増やすこと。deep model は、GBDT を置き換えるのではなく alignment / uncertainty feature extractor として小さく試す。
