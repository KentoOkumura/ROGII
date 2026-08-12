# 設計

## アプローチ

exp264 corrected OOF viewer/by-well、exp274 raw CatBoost OOF/by-well、raw train/typewell、corrected Stage C v6 selector score/plot manifestをID/wellでjoinする。well別RMSE差とrow-level SSE差をdistance、relative tail、GR欠損、target residualで集計する。selectorは`pred_abs_error` primary-domain top1を中心に、candidate別row share、悪化severity別lift、well dominant share、switch/marginを集計する。`p_within10`とfixed-domainは補助readoutとし、hard top1を最終予測とみなさない。

追加の切替寄与readoutでは、corrected Stage C v6 candidate-long 45,407,868 rowsからprimary 11候補のrow-level top1を復元する。well内で前rowとtop1 codeが異なるrowをswitchとし、`±0/1/5/25/100 rows`の近傍と非近傍でStage D final−exp274のSSE差を集計する。さらに同一top1が続くrunを作り、run先頭で切り替えた新候補のactual SSEと、直前run候補を現在run全体で維持したcounterfactual actual SSEを比較する。これはhard selector pathへのactual-TVT oracle attributionであり、Stage D finalのcounterfactualではない。

候補選択品質readoutでは、同じcandidate-longのprimary 11候補について`pred_abs_error`最小をselector top1、`actual_abs_error`最小をoracle top1とする。rowごとにselected hard SSE、oracle hard SSE、exp274 SSE、Stage D SSEを計算し、`selection_regret = selected_sse - oracle_sse`と`stage_d_effect = stage_d_sse - selected_sse`へ分ける。さらに`stage_d_vs_exp274 = selected_vs_exp274 + stage_d_effect`の加法恒等式を厳密に検証し、悪化wellで「誤候補選択」が悪化を作ったのか、Stage Dがselected hard pathを悪化または緩和したのかを判定する。

## 実験範囲

- 対象実験: `exp300_exp264_vs_exp274_well_selector_readout`
- Route: `ml_model`
- 親実験: source `exp264_exp263_candidate_confidence_dual_selector`、comparison `exp274_catboost_final_regressor_swap_on_exp238`
- 変更する変数: OOF比較surface、well severity、distance/relative-tail bucket、selector candidate/objective/domain
- 追加診断変数: selector switch、switchからのrow距離、candidate run、new-vs-previous candidate run counterfactual
- 固定する変数: OOF row、truth、exp264 corrected Stage D v3 prediction、exp274 raw CatBoost prediction、corrected Stage C v6 selector score、候補集合、閾値`0/0.25/1/3/5 ft`

## 再現性設計

- seed policy: 集計はno RNG。補助cross-validationを残す場合は`random_state=42`固定、global RNG不使用。
- stochastic 処理の有無: なし。KMeans等の探索的クラスタは最終主根拠から外す。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成0。保存済みcandidate TVT/selector scoreを読み取るだけ。
- 並列処理と乱数の関係: 並列処理なし。CSV/Parquetはstable sortとID coverage guardを使う。
- CPU/GPU runtime と deterministic flags: local CPU readout。学習0、GPU0。first full readoutのKaggle実行は、全入力がローカルに揃うため今回は既存Kaggle生成物のlocal deterministic再集計として記録する。
- train cache / test feature regeneration の SHA 記録方針: source file SHA、exp274 gzip decompressed SHA、selector score logical content/source manifest SHA、生成CSV SHAをsummary JSONへ保存する。
- model manifest / prediction / submission SHA 記録方針: model生成0、prediction生成0、submission生成0。入力OOF SHAを記録し、生成物はdiagnostic CSV/JSON/PNGのみ。
- Kaggle package bootstrap 確認方針: pushしない。canonical notebookはlocal artifactsを表示するdiagnostic構成として作り、`run_on_push=false`を維持する。

## リスク

- リークリスク: selector scoreはcorrected Stage C v6 strict nested outer-validだけを使用する。target-derived特徴、actual candidate error、well悪化labelは診断後joinに限定し、router承認には使わない。
- counterfactualリスク: previous-candidate holdはactual TVTを使うoracle attributionでありdeployable policyではない。Stage D finalはhard top1を直接採用しないため、切替がhard pathを悪化させてもfinal悪化の因果証明とはしない。
- oracle選択リスク: actual error最小候補はtruthを使う上限診断で、selectorの実運用候補ではない。selection regretは候補集合内ranking lossを測るが、候補集合外のexp274やStage Dの学習因果を直接証明しない。
- CV/LB 不一致リスク: exp264はPublic LB 7.562でexp274 7.715を改善した一方、cross-experiment OOFは悪化する。readoutはanchor更新・提出判断に使わない。
- ランタイム/メモリリスク: 45,407,868-row candidate-longを全件pandas保持しない。Parquet/CSVをchunk/column projectionで集計する。
- 再現性リスク: exp264/exp274の63 wellはouter fold assignmentが異なる。matched subsetを併記し、因果ablationと主張しない。

## 次の判断

exp300単独ではcandidate除外、hard fallback、threshold gridを行わない。既存exp276の固定risk familyが今回の高confidence・低switch regimeをfold-stableに捉える場合だけ、別の事前登録された安全化判断へ進む。
