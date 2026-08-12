# 設計

## アプローチ

exp035 の public sel15 inference flow を土台に、見えない test well 用処理 を exp038 selected single-LGBM に差し替える。notebook 内で exp029 train well の途中以降を隠した疑似 test 生成物 を読み、`base_plus_pf_prediction` feature set で final LightGBM residual model を fit する。見えない test wells には public sel15 PF feature generation と selector を実行し、single-LGBM prediction に fixed bucket shrink を適用する。

## 実験範囲

- 対象実験: `exp039_ravaghi_single_lgbm_inference_submit`
- Route: `ml_model`
- 親実験: `exp038_ravaghi_public_sel15_features_single_lgbm`
- 変更する変数: 見えない test well 用処理の predictor
- 固定する変数: public visible physical branch、PF/Beam/selector generation、offline execution

## リスク

- リークリスク: exp029 の train well の途中以降を隠した疑似 test 生成物の target は training label のみに使い、見えない test well の target は読まない。
- CV/LB 不一致リスク: exp038 selected candidate は public PF controls より弱いため、提出は要求対応として行い、採用候補とは限らない。
- ランタイム/メモリリスク: inference notebook 内で PF 128 seeds と LightGBM final fit を行うため、Kaggle runtime を監視する。
