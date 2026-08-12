# 設計

## アプローチ

1. train CSV から well-level metadata を再計算する。
   - `centroid_x/y`, `start/end`, `signed_azimuth_deg`, `eval_length`, `prefix_length`, `GR coverage`, `median_known_tvt` を作る。
   - typewell CSV の SHA16 から exact typewell group を作る。
2. 公式 PPT `AI_wellbore_geology_prediction_task_en.pptx` を zip として読み、slide10 の埋め込み画像を標準ライブラリだけで PNG decode する。
3. 赤い Verification well component を抽出し、plot 内正規化座標 `x_norm/y_norm` に変換する。
   - 抽出できない場合は deterministic grid fallback を使い、summary に fallback 理由を残す。
4. train wells の centroid を同じく 0-1 正規化し、PPT red component への最近傍距離を計算する。
5. target count 200 wells を目安に、red component を round-robin する greedy selection で holdout を作る。
   - `verification_like_spatial`: 空間分布を優先する通常版。
   - `verification_like_typewell_purged`: valid の exact typewell group と同じ group の train wells を `purged_train_excluded` にする厳しめ版。
6. 後続の anchor readout 用に `holdout_wells.csv`、`fold_assignments.csv`、`distribution_report.csv`、`summary.json` を保存する。

## 実験範囲

- 対象実験: `exp115_hidden_like_spatial_holdout_from_ppt`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 診断親: `exp044_stratified_groupkfold_cv_audit`, `exp065_typewell_supertype_cluster_cv_audit`, `exp073_gpu_reproducibility_guard_for_exp063_full_replay`, `exp098_selector_rank_slot_features_on_exp073`
- 変更する変数: holdout well selection surface
- 固定する変数: raw train CSV、official PPT、target-free well metadata、exact typewell SHA grouping

## 再現性設計

- seed policy: 乱数なし。well_id、PPT distance、component order の deterministic sort のみ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: 単一プロセス。global RNG は使わない。
- CPU/GPU runtime と deterministic flags: CPU only。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: PPTX SHA、slide image SHA、生成 CSV path、PPT red component count を summary に残す。
- model manifest / prediction / submission SHA 記録方針: モデル、予測、提出は作らない。deterministic submission anchor ではない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --notebook train --strict` で `config.yaml` と補助 `.py` を bootstrap に埋め込む。

## リスク

- リークリスク: holdout 作成では TVT true value を selection に使わない。後続 scoring では valid true TVT を feature / prior source に入れない。
- CV/LB 不一致リスク: PPT 由来 split は hidden split の復元ではないため、提出判断の唯一根拠にしない。
- ランタイム/メモリリスク: 773 wells と PPT PNG の処理のみ。Kaggle CPU で軽量。
- 再現性リスク: PPT の赤 component 抽出閾値に依存する。component count と fallback status を記録し、閾値変更時は別 version として扱う。
