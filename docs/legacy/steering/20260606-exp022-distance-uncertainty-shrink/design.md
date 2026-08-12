# 設計

## アプローチ

`exp021` を親にして、selected training variant `near_down_far_up_lightgbm` を固定する。各 fold では train-fold wells だけで weighted LightGBM residual model を fit し、valid wells の evaluation zone に OOF raw prediction を作る。

後処理は `last_anchor + alpha * residual` の形を維持する。baseline は exp014/exp021 の `distance_bucket_shrink` alpha を使う。uncertainty shrink は、その base alpha に inference-safe proxy から作る shrink factor を掛ける。

proxy:

- `distance`: hidden tail 開始からの row distance。
- `tail_progress`: hidden tail 内の進捗。
- `gr_missing`: evaluation row の GR 欠損。
- `z_span`: last known Z からの変位。
- `raw_residual_abs`: weighted model が last_anchor から離れた大きさ。

初回は conservative / medium / aggressive の固定 3 候補を比較する。target residual から係数を fit する場合は次実験で original-fold 外 selection として扱う。

## 実験範囲

- 対象実験: `exp022_distance_uncertainty_shrink`
- 親実験: `exp021_distance_weighted_inference_postprocess`
- 変更する変数: postprocess candidate method と uncertainty shrink 固定パラメータ。
- 固定する変数: LightGBM estimator、feature set `no_gr_signal`、sample weight profile `near_down_far_up_lightgbm`、GroupKFold by well、distance bucket shrink baseline。

## リスク

- リークリスク: shrink proxy は inference-time features だけに限定する。target residual を使った同一 OOF fit は clean CV として記録しない。
- CV/LB 不一致リスク: `exp021` は clean CV 改善にもかかわらず Public LB が悪化したため、CV 改善だけで LB anchor を更新しない。Public LB anchor は exp013 の 12.271 を維持する。
- ランタイム/メモリリスク: `exp021` と同じ weighted OOF flow なので train OOF は大きい。大きな `weighted_oof_predictions.csv` は Kaggle output / `/tmp` 側に置き、必要な summary と小さい metrics だけを repo に残す。
