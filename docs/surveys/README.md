# 調査レポート

完了した調査結果を探すときは、最初にこのファイルを参照します。対象は、実験構成・モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、論文・公開Notebook調査です。

## 保存ルール

- 人間が読む完了レポートは `docs/surveys/*.md` を正とします。
- 調査コードと生の表・図は `studies/`、実験実装・実行記録・公式結果は `experiments/` に残し、レポートからリンクします。
- 同じテーマの追調査は、原則として既存レポートを更新します。新しい問い・証拠範囲・結論になる場合だけ新しいレポートを作ります。
- `summary.md` は横断的な短い要約です。個別レポートを探す用途では、このREADMEの索引を使います。
- レポート本文を追加・更新したら、以下の「作成・完了手順」に従います。

## 作成・完了手順

1. この索引を検索し、同じ問いなら既存レポートを更新します。
2. 新しい問いだけ、次のようにレポートを作成します。このコマンドは`status: draft`、`summary: TODO`のファイルを作り、索引も一度更新します。

```bash
task new-survey-report \
  SURVEY_TITLE="exp238 selectorのモデル構成とOOF分析" \
  SURVEY_SLUG="exp238-selector-model-oof" \
  EXTRA_ARGS="--type experiment_review --type model_explanation --type oof_analysis --experiment exp238 --topic selector --topic confidence"
```

3. 本文を完成させ、`TODO`を除去し、一行`summary`を記入して`status: final`へ変更します。`task validate-template`と`task validate-config`は索引に含まれるdraftを許可しますが、完了判定の`task validate-surveys`はdraftに対して実行しません。
4. 完了後に索引を更新して検証します。

```bash
task update-survey-index
task validate-surveys
```

メタデータの `types`、`experiments`、`topics` は複数指定できます。実験番号は検索を安定させるため `exp238` の短い形式に統一します。

主な`types`は`experiment_review`、`model_explanation`、`oof_analysis`、`feature_analysis`、`comparison`、`survey`、`literature_review`です。必要な種類は組み合わせて使い、分類のためだけにレポートを分割しません。`task validate-surveys`はdraftまたは`TODO`が残っていると失敗します。

[横断調査サマリー](summary.md)

<!-- BEGIN AUTO SURVEY INDEX -->
## レポート一覧

| 日付 | レポート | 種類 | 実験 | トピック | 状態 | 一行要約 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-12 | [ROGII方針文書の移行前スナップショット](rogii-direction-migration-snapshot_20260812.md) | `strategy` | - | `history` | `final` | 2026-08-12の分割移行直前のKAGGLE_DIRECTION.md全文を保存した。 |
| 2026-08-09 | [ROGII戦略判断履歴](rogii_strategy_history_20260809.md) | `strategy` | - | `history`, `strategy` | `final` | KAGGLE_DIRECTION.mdから退避した、2026-08-09までの実験横断の判断履歴。 |
| 2026-08-09 | [ROGII実験横断メモ履歴](experiment_summary_history_20260809.md) | `comparison` | - | `history`, `experiment_summary` | `final` | 旧experiment_summary.mdから退避した、主な発見と変更履歴の手書き記録。 |
| 2026-08-06 | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) | `survey`, `literature_review`, `comparison` | `exp179`, `exp182`, `exp202`, `exp210`, `exp212`, `exp215`, `exp223`, `exp235`, `exp413`, `exp512` | `winning_solution`, `agent_workflow`, `prompt_design`, `skill_design`, `candidate_path`, `validation_shift`, `blind_evaluation`, `idea_generation` | `final` | 最終上位8解法を比較し、source-hidden blind benchmark 2回でtop 5の15/16・全12案の16/16機構再発見を確認したkaggle-idea-forgeを実装した。 |
| 2026-08-04 | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `comparison` | `exp357`, `exp413`, `exp490` | `hmm`, `mean_reversion`, `reuse`, `risk`, `ensemble` | `final` | exp490の平均回帰機構、OOFでの利点とtailリスクを整理し、直接blendではなく証拠・risk特徴としての再利用方針を定めた。 |
| 2026-07-19 | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `feature_analysis` | `exp263`, `exp264` | `selector`, `candidate_path`, `feature_importance`, `hidden_safe`, `lightgbm` | `final` | exp264の12候補bank、hidden-safe selector、compact特徴、最終TVTモデルとguard結果を統合して説明する。 |
| 2026-07-16 | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) | `oof_analysis`, `comparison` | `exp072`, `exp103`, `exp104`, `exp192`, `exp209`, `exp221`, `exp223`, `exp226`, `exp232`, `exp233`, `exp234`, `exp240`, `exp243` | `candidate_path`, `blend`, `pf_beam`, `hmm`, `selector` | `final` | 保存OOF上の候補パスを横断比較し、exp226とlikPF・exact HMM系の固定結合を主要候補として整理した。 |
| 2026-07-16 | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `feature_analysis`, `comparison` | `exp148`, `exp218`, `exp226`, `exp237`, `exp238`, `exp243`, `exp251`, `exp257` | `selector`, `candidate_path`, `feature_importance`, `confidence`, `lightgbm` | `final` | exp238の候補bank、nested selector、415列TVTモデル、OOF安全性と特徴量重複を統合して説明する。 |
| 2026-07-16 | [Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) | `model_explanation`, `feature_analysis`, `comparison` | `exp237`, `exp238`, `exp251` | `selector`, `feature_catalog`, `feature_importance`, `confidence` | `final` | exp237・exp251のselector入力とexp238 downstream adapterを区別し、全特徴の意味と重要度をカタログ化した。 |
| 2026-07-12 | [HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) | `oof_analysis`, `comparison` | `exp072`, `exp073`, `exp083`, `exp209`, `exp223`, `exp226` | `error_analysis`, `hmm`, `pf_beam`, `well` | `final` | HMM、PF、exp226のwell別失敗を比較し、whole-well offset、長いtail、GR・geometryの役割差と候補併用条件を整理した。 |
| 2026-07-07 | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) | `survey`, `literature_review`, `comparison` | `exp082`, `exp179`, `exp182`, `exp193`, `exp204`, `exp206`, `exp214`, `exp215` | `external_data`, `public_artifact`, `data_generation`, `hidden_safe`, `licensing` | `final` | 外部データ、公開生成物、コンペ内候補生成を比較し、hidden-safe性とライセンスを踏まえた利用優先度を整理した。 |
| 2026-07-05 | [Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md) | `oof_analysis`, `comparison` | `exp001` | `prefix_extrapolation`, `tail`, `candidate_path` | `final` | known prefixからの直接外挿はlong tailで不安定であり、小さいblendまたは信頼度特徴として限定利用する判断を記録した。 |
| 2026-06-25 | [GR matching deep research](gr_matching_deep_research_20260625.md) | `survey`, `experiment_review`, `comparison` | `exp008`, `exp017`, `exp042`, `exp048`, `exp091`, `exp093`, `exp099`, `exp120`, `exp223` | `gr_matching`, `typewell`, `candidate_path`, `observation_likelihood` | `final` | GR matchingの外部知見と既存実験を統合し、直接TVT決定ではなく候補confidence・likelihoodとして使う方向を残した。 |
| 2026-06-20 | [PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md) | `oof_analysis`, `comparison` | `exp073`, `exp083` | `error_analysis`, `pf_beam`, `candidate_path` | `final` | PF・Beam・MLを773 wellsで比較し、PF直接置換を否定する一方、disagreementを候補選択やsample weightの診断材料として残した。 |
| 2026-06-16 | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) | `oof_analysis`, `comparison` | `exp026`, `exp027`, `exp039`, `exp054`, `exp063`, `exp069`, `exp070`, `exp073` | `error_analysis`, `tail`, `candidate_path` | `final` | visible sampleで候補予測を比較し、global blendではなくwell geometryに応じた限定的な分岐だけを後続候補とした。 |
| 2026-05-28 | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) | `survey`, `literature_review` | - | `well_log_correlation`, `geosteering`, `gr_matching`, `formation_surface`, `sequence_model` | `final` | well-log相関、geosteering、formation推定などの関連研究を整理し、ROGIIへ転用可能な候補と検証上の注意をまとめた。 |

## 実験番号別

| キー | レポート |
| --- | --- |
| `exp001` | [Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md) |
| `exp008` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp017` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp026` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp027` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp039` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp042` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp048` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp054` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp063` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp069` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp070` | [visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp072` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) |
| `exp073` | [HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `exp082` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp083` | [HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md) |
| `exp091` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp093` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp099` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp103` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp104` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp120` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp148` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `exp179` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp182` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp192` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp193` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp202` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp204` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp206` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp209` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) |
| `exp210` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp212` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp214` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp215` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp218` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `exp221` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp223` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp226` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) |
| `exp232` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp233` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp234` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp235` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp237` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `exp238` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `exp240` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp243` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `exp251` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `exp257` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `exp263` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md) |
| `exp264` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md) |
| `exp357` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `exp413` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `exp490` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `exp512` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |

## 種類別

| キー | レポート |
| --- | --- |
| `comparison` | [ROGII実験横断メモ履歴](experiment_summary_history_20260809.md)<br>[ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `experiment_review` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `feature_analysis` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `literature_review` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `model_explanation` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `oof_analysis` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `strategy` | [ROGII方針文書の移行前スナップショット](rogii-direction-migration-snapshot_20260812.md)<br>[ROGII戦略判断履歴](rogii_strategy_history_20260809.md) |
| `survey` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |

## トピック別

| キー | レポート |
| --- | --- |
| `agent_workflow` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `blend` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `blind_evaluation` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `candidate_path` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `confidence` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `data_generation` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `ensemble` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `error_analysis` | [HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `experiment_summary` | [ROGII実験横断メモ履歴](experiment_summary_history_20260809.md) |
| `external_data` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `feature_catalog` | [Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `feature_importance` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `formation_surface` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `geosteering` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `gr_matching` | [GR matching deep research](gr_matching_deep_research_20260625.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `hidden_safe` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `history` | [ROGII方針文書の移行前スナップショット](rogii-direction-migration-snapshot_20260812.md)<br>[ROGII戦略判断履歴](rogii_strategy_history_20260809.md)<br>[ROGII実験横断メモ履歴](experiment_summary_history_20260809.md) |
| `hmm` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) |
| `idea_generation` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `licensing` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `lightgbm` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `mean_reversion` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `observation_likelihood` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `pf_beam` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md)<br>[PF・Beam disagreementとwell別誤差の監査](pf_beam_disagreement_error_map_20260620.md) |
| `prefix_extrapolation` | [Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md) |
| `prompt_design` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `public_artifact` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `reuse` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `risk` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `selector` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `sequence_model` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `skill_design` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `strategy` | [ROGII戦略判断履歴](rogii_strategy_history_20260809.md) |
| `tail` | [Prefix Extrapolation Reasonableness Audit](prefix_extrapolation_reasonableness_20260705.md)<br>[visible tailの候補別・well別誤差監査](metric_weighted_tail_error_map_20260616.md) |
| `typewell` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `validation_shift` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `well` | [HMM・PF・exp226のwell別失敗パターン監査](hmm_pf_exp226_well_pattern_readout_20260712.md) |
| `well_log_correlation` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `winning_solution` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
<!-- END AUTO SURVEY INDEX -->
