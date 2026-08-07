# 調査レポート

完了した調査結果を探すときは、最初にこのファイルを参照します。対象は、実験構成・モデル説明、OOF／結果EDA、特徴量・failure mode、複数実験比較、論文・公開Notebook調査です。

## 保存ルール

- 人間が読む完了レポートは `docs/surveys/*.md` を正とします。
- 調査コードと生の表・図は `studies/`、実験実装・実行記録・公式結果は `experiments/` に残し、レポートからリンクします。
- 同じテーマの追調査は、原則として既存レポートを更新します。新しい問い・証拠範囲・結論になる場合だけ新しいレポートを作ります。
- `summary.md` は横断的な短い要約です。個別レポートを探す用途では、このREADMEの索引を使います。
- レポート本文を追加・更新したら `task update-survey-index`、CI相当の確認では `task validate-surveys` を実行します。

新規作成例:

```bash
task new-survey-report \
  SURVEY_TITLE="exp238 selectorのモデル構成とOOF分析" \
  SURVEY_SLUG="exp238-selector-model-oof" \
  EXTRA_ARGS="--type experiment_review --type model_explanation --type oof_analysis --experiment exp238 --topic selector --topic confidence"
```

メタデータの `types`、`experiments`、`topics` は複数指定できます。実験番号は検索を安定させるため `exp238` の短い形式に統一します。

主な`types`は`experiment_review`、`model_explanation`、`oof_analysis`、`feature_analysis`、`comparison`、`survey`、`literature_review`です。必要な種類は組み合わせて使い、分類のためだけにレポートを分割しません。作業中は`status: draft`を使えますが、`task validate-surveys`はdraftが残っていると失敗します。

[横断調査サマリー](summary.md)

<!-- BEGIN AUTO SURVEY INDEX -->
## レポート一覧

| 日付 | レポート | 種類 | 実験 | トピック | 状態 | 一行要約 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-06 | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) | `survey`, `literature_review`, `comparison` | `exp179`, `exp182`, `exp202`, `exp210`, `exp212`, `exp215`, `exp223`, `exp235`, `exp413`, `exp512` | `winning_solution`, `agent_workflow`, `prompt_design`, `skill_design`, `candidate_path`, `validation_shift`, `blind_evaluation`, `idea_generation` | `final` | 最終上位8解法を比較し、source-hidden blind benchmark 2回でtop 5の15/16・全12案の16/16機構再発見を確認したkaggle-idea-forgeを実装した。 |
| 2026-08-04 | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `comparison` | `exp357`, `exp413`, `exp490` | `hmm`, `mean_reversion`, `reuse`, `risk`, `ensemble` | `final` | exp490の平均回帰機構、OOFでの利点とtailリスクを整理し、直接blendではなく証拠・risk特徴としての再利用方針を定めた。 |
| 2026-07-19 | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `feature_analysis` | `exp263`, `exp264` | `selector`, `candidate_path`, `feature_importance`, `hidden_safe`, `lightgbm` | `final` | exp264の12候補bank、hidden-safe selector、compact特徴、最終TVTモデルとguard結果を統合して説明する。 |
| 2026-07-16 | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) | `oof_analysis`, `comparison` | `exp072`, `exp103`, `exp104`, `exp192`, `exp209`, `exp221`, `exp223`, `exp226`, `exp232`, `exp233`, `exp234`, `exp240`, `exp243` | `candidate_path`, `blend`, `pf_beam`, `hmm`, `selector` | `final` | 保存OOF上の候補パスを横断比較し、exp226とlikPF・exact HMM系の固定結合を主要候補として整理した。 |
| 2026-07-16 | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) | `experiment_review`, `model_explanation`, `oof_analysis`, `feature_analysis`, `comparison` | `exp148`, `exp218`, `exp226`, `exp237`, `exp238`, `exp243`, `exp251`, `exp257` | `selector`, `candidate_path`, `feature_importance`, `confidence`, `lightgbm` | `final` | exp238の候補bank、nested selector、415列TVTモデル、OOF安全性と特徴量重複を統合して説明する。 |
| 2026-07-16 | [Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) | `model_explanation`, `feature_analysis`, `comparison` | `exp237`, `exp238`, `exp251` | `selector`, `feature_catalog`, `feature_importance`, `confidence` | `final` | exp237・exp251のselector入力とexp238 downstream adapterを区別し、全特徴の意味と重要度をカタログ化した。 |
| 2026-07-07 | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) | `survey`, `literature_review`, `comparison` | `exp082`, `exp179`, `exp182`, `exp193`, `exp204`, `exp206`, `exp214`, `exp215` | `external_data`, `public_artifact`, `data_generation`, `hidden_safe`, `licensing` | `final` | 外部データ、公開生成物、コンペ内候補生成を比較し、hidden-safe性とライセンスを踏まえた利用優先度を整理した。 |
| 2026-06-25 | [GR matching deep research](gr_matching_deep_research_20260625.md) | `survey`, `experiment_review`, `comparison` | `exp008`, `exp017`, `exp042`, `exp048`, `exp091`, `exp093`, `exp099`, `exp120`, `exp223` | `gr_matching`, `typewell`, `candidate_path`, `observation_likelihood` | `final` | GR matchingの外部知見と既存実験を統合し、直接TVT決定ではなく候補confidence・likelihoodとして使う方向を残した。 |
| 2026-05-28 | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) | `survey`, `literature_review` | - | `well_log_correlation`, `geosteering`, `gr_matching`, `formation_surface`, `sequence_model` | `final` | well-log相関、geosteering、formation推定などの関連研究を整理し、ROGIIへ転用可能な候補と検証上の注意をまとめた。 |

## 実験番号別

| キー | レポート |
| --- | --- |
| `exp008` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp017` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp042` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp048` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp072` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp082` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
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
| `exp209` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp210` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp212` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `exp214` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp215` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `exp218` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `exp221` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `exp223` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `exp226` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
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
| `comparison` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `experiment_review` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `feature_analysis` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `literature_review` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `model_explanation` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `oof_analysis` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `survey` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |

## トピック別

| キー | レポート |
| --- | --- |
| `agent_workflow` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `blend` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `blind_evaluation` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `candidate_path` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md)<br>[exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[GR matching deep research](gr_matching_deep_research_20260625.md) |
| `confidence` | [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `data_generation` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `ensemble` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `external_data` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `feature_catalog` | [Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `feature_importance` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `formation_surface` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `geosteering` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `gr_matching` | [GR matching deep research](gr_matching_deep_research_20260625.md)<br>[関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `hidden_safe` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `hmm` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `idea_generation` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `licensing` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `lightgbm` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md) |
| `mean_reversion` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `observation_likelihood` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `pf_beam` | [候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md) |
| `prompt_design` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `public_artifact` | [ROGII external data and data generation survey](rogii_external_data_and_generation_20260707.md) |
| `reuse` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `risk` | [exp490 再利用戦略調査](exp490_reuse_strategy_20260804.md) |
| `selector` | [exp264 selector / TVT feature audit](exp264_selector_tvt_feature_audit_20260719.md)<br>[候補パス平均・凸結合の全横断監査](candidate_path_blend_audit_20260716.md)<br>[exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)<br>[Selector入力特徴量カタログと重要度](selector_feature_catalog_20260716.md) |
| `sequence_model` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `skill_design` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `typewell` | [GR matching deep research](gr_matching_deep_research_20260625.md) |
| `validation_shift` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
| `well_log_correlation` | [関連研究調査 ROGII Wellbore Geology Prediction](maybe_related_research.md) |
| `winning_solution` | [ROGII 上位解法と agent-driven Kaggle 着想ワークフロー](rogii-top-solutions-agent-idea-workflow_20260806.md) |
<!-- END AUTO SURVEY INDEX -->
