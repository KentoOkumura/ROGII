# 設計

## アプローチ

exp287のformation add-onlyが作ったglobal改善とwell-level regressionを、保存済みOOFとfold-safe formation cacheだけで原因分解する。モデルの特徴量重要度やworst-well後付け規則ではなく、hidden testでも生成可能なformation reliability属性を誤差なしで先に凍結し、その後にexp287とcorrected exp264のwell別OOF差分を評価する。

これは予測改善実験ではない。安定した事前risk familyが存在するかを反証し、次の単一変更trainを設計する資格があるかだけを決める。

## 仮説

exp287のtail悪化がformation referenceの外挿距離、不確実性、plane/dense不一致、formation consensus不足、known-prefix calibration不良のいずれかに由来するなら、そのtarget-free属性の高risk四分位では低risk四分位より`RMSE(exp287)-RMSE(exp264)`が大きく、4/5 foldsと2 hidden-like面で同方向に再現する。

## 実験範囲

- 対象実験: `exp336_exp287_formation_tail_attribution_readout`
- Route: `ml_model`診断
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 比較対象: corrected `exp264_exp263_candidate_confidence_dual_selector`
- trigger: `exp334_equal_well_loss_weighting_on_exp287`
- 変更する変数: なし。target-free属性を診断表示するだけ。
- 固定する変数: exp287/exp264 OOF、outer folds、score rows、formation valid cache、feature schema、hidden-like assignment、主endpoint、6 family、aggregation、四分位、risk方向、coverage、gate。
- 除外: exp334 OOF scoring、model fit、feature deletion、補正prediction、inference、submission。

## 根拠anchor

| 実験 | CV | Public LB | tail判断 |
| --- | ---: | ---: | --- |
| corrected exp264 | 8.460811238 | 7.562 | 比較基準 |
| exp287 | 8.136708220 | 7.530 | global/LB改善、worst `+8.228410 ft`でtrain guard FAIL |
| exp334 | 8.093497524 | 未提出 | exp287比global改善、worst `+7.156485 ft`でtail FAIL |

exp287対exp264はformation 74 add-onlyの比較であり、formation attributionの主endpointとして使える。exp334はlossを変更するため、readout開始条件の根拠に限定する。

## 入力契約

### exp287

- OOF: `fold_safe_formation_oof_predictions.parquet`
- OOF SHA: `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- model manifest SHA: `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- metrics SHA: `435434342494aaa62cee6e627809363ac34f16174973f4b81301d2923f780862`
- fold metrics SHA: `864eca0452eea578c96baa653d25c4f2ae241c84b8e5d659b277407b5e427141`
- by-well SHA: `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- formation fold manifest SHA: `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- full 421-model schema SHA: `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`
- formation 74-cache schema SHA: `64e8ceb0e0cf63317d040d7d72bbc8beae4a8805ba9a189dec04888abfcd914f`
- manifest全10 partitionを検証し、Stage Aは各foldの`role=valid` 5 partitionだけを読む。

### corrected exp264

- OOF: `stage_d_oof_predictions.parquet`
- OOF SHA: `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- prediction: `selector_compact_addonly__lgb_mean__pred_tvt`

### 共通

- rows / wells / folds: `3,783,989 / 773 / 5`
- hidden-like assignment SHA: `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`
- raw target-free context: horizontal CSVの`MD/X/Y/Z/TVT_input`のみ。
- ID、well、fold、actual TVTをStage Bで完全照合する。actual TVT toleranceは`1e-4 ft`。

## Stage A: target-free attribute freeze

### 読み込み境界

Stage Aはformation valid cacheの`id/well/74 features`、manifestのfold/reference evidence、raw contextの`MD/X/Y/Z/TVT_input`だけを読む。`TVT`、target、actual、prediction、error、abs/squared error、worst-well ID、by-well outcomeはpath解決・schema・関数引数の時点で禁止する。

5 valid partitionsを結合し、全OOF IDがexactly once、773 wellsが各1 outer fold、74 featuresがfinite、manifest file/logical/schema SHA一致であることを確認する。

### Primary risk family

| Family | row source | well scalar | risk方向 |
| --- | --- | --- | --- |
| `plane_reference_distance` | `spatial_knn_dist` | p90 | 高いほどrisk |
| `dense_reference_distance` | `dense_dist` | p90 | 高いほどrisk |
| `dense_neighbor_uncertainty` | `dense_std` | p90 | 高いほどrisk |
| `plane_dense_disagreement` | `abs(spatial_vs_dense)` | p90 | 高いほどrisk |
| `formation_consensus_spread` | `form_rng_d` | p90 | 高いほどrisk |
| `known_prefix_formation_calibration_error` | `frm_rmse_*` 6列と`dense_rmse` | well定数7値のmax | 高いほどrisk |

row quantileとpopulation quartileはNumPy `method="linear"`を使う。well scalarを773 wellsで集計し、q25/q50/q75を保存する。edgeがstrictに増加しないfamilyはtechnical ineligibleとし、rank binや別transformで救済しない。

### Context / falsification readout

次は原因分解の補助表示に限り、primary PASS familyにはしない。

- fold別plane/dense reference利用可能well数
- formation cacheのnonfinite/missing rate。設計上0を必須とする。
- known-prefix row数、evaluation/known row比
- suffix/prefix MD span比
- evaluation点からlast-known XYまでのp90距離 / prefix XY span
- `sig_std` p90、`dense_nb_std`

generic trajectory contextだけが再現し、6 formation familyが通らない場合はformation attributionを支持しない。必要なら別のgeneric geometry仮説として新規設計する。

### Freeze barrier

- `target_free_well_attributes.csv`を`well`順、列を契約順で保存する。
- family、scalar、risk方向、eligible、q25/q50/q75、coverageをfreeze manifestへ保存する。
- CSV logical contentとmanifest canonical JSONのSHA256を計算する。
- error-related module/file/pathをopenしていない証拠と禁止列auditを保存する。
- Stage Bはfreeze manifest SHAをconfig/実行状態へ明示してからのみ開始できる。

## Stage B: frozen attribution

1. exp287/exp264 OOFをSHA検証し、ID/well/fold/actual TVTを照合する。
2. 各wellについてscore rowsの非加重RMSEを計算する。
3. `delta_w = RMSE_exp287,w - RMSE_exp264,w`を主endpointとする。wellは等重み。
4. Stage Aの固定境界だけでQ1–Q4へ割り当てる。
5. familyごとにglobal、fold、hidden-like spatial、hidden-like typewell-purgedのQ4−Q1 mean/median deltaとcoverageを出す。
6. `+1/+3/+5 ft`悪化率、riskとのSpearman、row-weighted RMSE、最大absolute-delta well除外感度はreport-onlyとし、PASS判定へ使わない。

worst-well IDはmetrics出力で観測結果として表示できるが、属性、bin、family、gateの作成には使わない。

## 固定decision gate

familyごとに次をAND判定する。

1. global Q4−Q1 mean delta `>= +0.25 ft`。
2. global Q4−Q1 median delta `> 0`。
3. Q4−Q1 mean deltaが正のfold `>=4/5`。
4. hidden-like spatialとtypewell-purgedのQ4−Q1 mean deltaが両方正。
5. Q1/Q4 coverageがglobal各100 wells、fold各10 wells、hidden-like各15 wells以上。
6. family eligible、error非依存、固定方向/境界、全technical audit PASS。

`any(primary family PASS)`なら`ATTRIBUTION_SUPPORTED`、0件なら`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`とする。

- `ATTRIBUTION_SUPPORTED`: 別の単一変更介入実験を設計できる。どの介入にするかはユーザー確認事項であり、本実験では決めない。
- `NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`: formation reliability救済枝を閉じ、同じOOFでfamily/thresholdを追加しない。

PASSはexp287/exp334のtrain-side昇格、inference、submissionを許可しない。

## 生成物契約

- `scientific_contract.json`
- `input_artifact_manifest.json`
- `target_free_well_attributes.csv`
- `target_free_attribute_freeze_manifest.json`
- `well_oof_delta_metrics.csv`
- `family_quartile_metrics.csv`
- `fold_direction_metrics.csv`
- `hidden_like_direction_metrics.csv`
- `technical_context_readout.csv`
- `attribution_decision.json`
- `reproducibility_manifest.json`

corrected prediction、OOF、model、submissionは生成しない。

## 実行量

- primary families: 6
- model variants: 0
- LightGBM configs: 0
- trained folds: 0
- boosters: 0
- control retraining: 0
- PF/HMM/Beam runs: 0
- runtime: Kaggle CPU、single worker、BLAS thread 1、internet off。preflight上限14,400秒。version 2 readout本体は92.458秒で完了した。

## 再現性設計

- seed policy: RNGなし。
- stochastic処理: なし。
- 並列と乱数: single worker、BLAS thread 1。乱数なし。
- canonical order: attributeはwell/family、metricsはfamily/scope/fold/quartile、JSON keyはsortする。
- input SHA: exp287/exp264/hidden-like成果物をfile SHAで固定する。
- feature content SHA: Stage A CSVをcanonical row/column orderのlogical content SHAで固定する。
- freeze barrier: freeze manifest SHAをStage B開始前に記録する。
- model/prediction/submission SHA: 非該当。
- Kaggle bootstrap: 実装・push承認後に正のconfig/sourceから再生成し、approval、stage、input SHA、family/gate、CPU/internet設定を展開後configと照合する。
- deterministic anchor: submission anchorではない。固定入力に対するdiagnostic reproducibilityだけを主張する。

## リスク

- Leakage: errorを見て属性やboundaryを作るとsame-OOF overfitになる。Stage A/Bのload境界とfreeze SHAをhard gateにする。
- 多重比較: 6 familyのどれか1件を選ぶため偶然PASSがあり得る。practical effect`0.25 ft`、median、4/5 folds、2 hidden-like面、coverageをすべて要求する。
- Confounding: 長いsuffixやtrajectory難度がformation riskと相関し得る。context readoutで表示するがposthoc調整は行わない。formation family不通過ならgeneric geometryを代用PASSにしない。
- CV/LB: Public testは3 wellsで、Public LB 7.530はtail safetyを保証しない。本readoutは773-well OOFとhidden-likeを根拠にし、LBへ最適化しない。
- Artifact: exp287 formation cachesは大きい。manifestからvalid 5 partitionだけを読むが、実装時に必要なKaggle source mountとruntime preflightを確認する。
- 再現性: Parquet物理byte差ではなくmanifest file SHAとlogical content SHAを区別する。

## 実行結果

- Kaggle private CPU kernel: `kentookumura/exp336-exp287-formtail-attribution-readout-train`
- id_no / completed version: `128221753 / 2`
- Stage A freeze manifest SHA: `e65a9924c11f77008d1574070f71b6cf2d099993e8510eeaf7cc285c5d54979f`
- 全6 familyがstrict edge、error-independent boundary、coverageを通過した。
- passed familyは`0/6`。固定decisionは`NO_STABLE_FORMATION_ATTRIBUTION_CLOSE`。
- `dense_reference_distance`はglobal効果量`+0.350471 ft`、median正、5/5 folds正だったが、hidden-like 2面がともに逆方向でFAILした。他familyも事前登録AND gateを満たさなかった。
- model/prediction/inference/submissionは生成していない。同じOOFでのfamily/threshold救済は行わず、formation attribution枝を閉じる。
- 運用: Late phaseであるため診断を肥大化させない。family追加や介入trainを同一expへ混ぜない。

## 次のアクション

compact self-contained Jupytext train候補とfail-closed inference候補を実装し、synthetic testsと静的検査まで完了した状態で停止する。既存canonical Notebookは明示採用前に上書きしない。Kaggle CPU package/push/runは別途承認を得る。
