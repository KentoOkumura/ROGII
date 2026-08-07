# exp264 selector 出力契約

> **無効化済み:** 現行Stage C schemaはtraining-only formation raw/delta 12特徴を含むため、既存score、
> compact、Stage D add-only OOFと本契約の推論出力を信頼しない。以下は失敗時のhistorical contractであり、
> raw-test-only schemaを再構築するまで実行契約として使用しない。

## 結論

selectorは12候補それぞれに`pred_abs_error`と`p_within10`を出す。この候補別scoreはtop1、top2、margin、entropy、確信度を計算する内部表現として必須である。一方、推論時にscore CSVを一度保存して読み直したり、Viterbiで最終pathを作ったりはしない。同じfold/process内で直ちにcompact meta-featureへ変換する。

## 1. 監査用candidate-long出力

OOFの評価と再現性監査のため、次をParquetで保存する。

- key: `id`, `well`, `well_row_idx`, `outer_fold`, `candidate_id`
- value: `candidate_tvt`, `candidate_available`, `confidence_valid`
- dual score: `pred_abs_error`, `p_within10`
- OOF label-only audit: `actual_abs_error`, `actual_within10`。feature schemaには含めない。
- provenance: `feature_schema_sha`, `model_sha`, `candidate_contract_sha`

これはselectorのcanonical audit artifactであり、TVTの最終予測ではない。current-testでは同じ情報をメモリ上で生成し、公開test行artifactへの依存を作らない。必要なら小さいparity sampleだけを保存する。

source-native confidenceを採用feature schemaが使う場合、exp263 Stage 1は
`confidence__<candidate_id>__<field>`形式でcurrent-test confidenceを出し、train cacheとの
意味・coverage parityを通す必要がある。未出力のnative confidenceを推論時に暗黙の0や全NaNへ
置き換えて実行しない。availability、anchor距離、shape、bank/formula disagreementだけで構成された
universal proxyは全候補・current-testで常に生成する。

Stage A採用schemaに対するcurrent-test必須namespaceは21列である。内訳はexp226 2列、self-GR
HMM 10列、likPFの明示valid 1列、exact HMM 4列、PF-ANCC 2列、Beam 2列。likPFの
`confidence_valid=False`は「候補が無効」ではなく「source-native scalarがない」ことを表し、候補値、
shape、bank disagreementは通常どおり有効である。exp264 inferenceは21列の存在、numeric finite、
native fieldを持つ5候補のvalid全行trueを確認してからfeatureを作る。

## 2. 運用用compact meta-feature

candidate-long scoreから決定的にwide化する。少なくとも次のgroupを持つ。

- 12候補別の`pred_abs_error`と`p_within10`
- `primitive_pair_bank` 11本内の2目的それぞれのtop1/top2候補値・score・margin
- `primitive_fixed_bank` 7本内のfixed fallback比較、top1、margin
- classifier/regressorのtop1一致
- top1候補と`last_known_tvt`との差
- 候補scoreのmean/std、`p_within10` entropy
- 候補値のrange/std、confidence-valid数、available数
- primary `pred_abs_error` top1の11 one-hot
- top1がprimitive/pair/fixedのどれかを示すflag

12候補すべてを一つのhard-selectable domainにしたtop1/marginは作らない。exp263のlineage guardに従い、pairを含む11本domainとfixedを含む7本domainを分ける。`blend_likpf_hmm_w500`は`likpf_mean__exact_hmm`のaliasなので独立slotを持たない。

列数は35へ固定しない。exp238の35列は11候補・単一objective用だったため、12候補・dual objectiveへ拡張した本実験ではStage Aで`compact_meta_schema.json`を生成しSHA固定する。top1やmarginを直接回帰するモデルは作らず、候補別の明確な教師labelから学習した2 scoreを決定的に変換する。

## 3. nested stacking

- outer-train: outer-train内のinner OOF selector scoreを即compact化する。
- outer-valid: outer-train内inner model ensembleだけでscoreを作り即compact化する。
- downstream TVT LightGBM: 漏洩のないcompact featureをadd-onlyで学習する。
- current-test: 保存済みselectorを適用して即compact化し、selectorもTVTモデルも再学習しない。

Stage Cのtrain出力は、後段outer foldごとのfeature provenanceを失わないよう次の25 partitionへ分ける。

- `nested_compact_meta/downstream_outer_fold=<0..4>/role=train/source_outer_fold=<fold>/part-00000.parquet`
- `nested_compact_meta/downstream_outer_fold=<0..4>/role=valid/source_outer_fold=<fold>/part-00000.parquet`
- 各downstream outer foldはtrain 4 partition + valid 1 partition。
- 合計base rowsは`3,783,989 × 5 = 18,919,945`。
- outer-valid監査scoreは各row一度だけ、12 candidate-longで`45,407,868` rows。

Stage Cは`nested_selector_model_manifest.json`、`nested_fold_manifest.csv`、
`nested_compact_partition_manifest.csv`、`nested_compact_manifest.json`、
`nested_outer_valid_candidate_score.parquet`、`nested_selector_metrics.json/csv`を保存する。
inner foldはouter-train wellだけからrow-balancedにdeterministic生成し、outer-valid wellをassignment、
fit、early stoppingへ一切入れない。

selector-only Stage B、nested Stage C、matched TVT control/add-onlyは計算上完走したが、Stage Aのraw-test availability監査漏れによりStage B/C score、compact、Stage D add-only比較をすべて無効化した。旧guard判定と7.805644は性能根拠にしない。

current-test推論出力は次を正とする。

- `current_test_formula_parity.parquet`: 同じrunでraw testから再生成した12候補と21 confidence列。
- `compact_meta_current_test_outer<0..4>.parquet`: Stage C outer別8 selector modelから生成した74列。TVTモデル入力にはメモリ上のframeを使い、保存物を再読込しない。
- `selector_missingness_current_test.csv`: 100 selector特徴の学習時/current-test欠損率、欠損数、構造的欠損flag。
- `selector_missingness_by_candidate_current_test.csv`: 12候補別のselector入力欠損数・欠損率。
- `stage_d_compact_current_test_predictions.csv.gz`: 15 add-only modelのfold/config別監査値と最終平均予測。
- `submission.csv`: sample submission順の`id,tvt`。生成のみでcompetition submitはしない。
- `inference_metrics.json` / `reproducibility_manifest_inference.json`: input/schema/model/prediction/submission SHA、runtime、guard override scopeを保存する。

selectorのraw 100列はStage A catalogで学習時欠損率を固定する。学習時から疎な29列のNaNは
LightGBMへそのまま渡し、0補完しない。`±inf`、training-dense列の新規NaN、`conf__`/`formula__`の
構造的欠損率ずれ、current-test全欠損化だけを停止条件とする。compact 74列と後段454列はfinite必須を維持する。

## 4. scope外

- HMM+LGB exp221/234/240
- selector scoreをHMM emissionへ戻すこと
- Viterbi/CRF/TCNによる最終path生成
- hard selectorの提出
- compact add-only TVTモデルの自動昇格、またはguard PASSとみなすこと
- Kaggle competitionへのsubmit操作
- candidate TVTのsoftmax平均
- exp251の旧11候補やexp263で除外された`sc_ens`, `hyb`, `tvt_dense*`の復活
