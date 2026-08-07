# 要件

## 依頼

`hmm_exp226_selector_rank_slot_addonly_on_exp218` を、厳密な nested stacking として実装する。

## 制約

- Route: `ml_model`。
- 親は `exp218_gr_wavelet_rotation_confidence_features_on_exp148`、selector 親は `exp237_hmm_exp226_candidate_selector_on_exp183`。
- outer GroupKFold の validation well は、selector と最終 LightGBM の全学習から除外する。
- outer-train の selector 特徴は inner GroupKFold OOF、outer-valid は inner models の平均 score から生成する。
- selected path の直接置換、blend、postprocess、residual anchor 化は禁止する。
- safety guard 前の Kaggle train push、inference、submitは禁止する。

## 受け入れ基準

- outer 5 folds / inner 4 folds の split contract をコードで検証する。
- near `000_050`、worst-well、global、`1000+`、hidden-like を selector safety audit に保存する。
- guard 通過時のみ selector rank-slot 特徴を exp218 surface に add-only する。
- active variant 1、selector 20 boosters、final 15 boosters、合計35 boostersを記録する。
- feature/model/prediction SHA を保存し、gzipはdecompressed SHAを主証拠にする。
- selector trainでouter 5 × inner 4の20モデル本体を保存し、推論では再学習しない。
- raw testはouter fold別の4 selector平均で5個のscore面を作り、対応するouter foldの最終LightGBMだけへ渡す。
- code submissionのhidden testではpublic-test行に紐づくexp073 cache、exp226 submission、exp145 rawtest特徴、selector score artifactを読まない。
- 提出notebook内でcurrent testからbase replay、HMM、K16、multi-observation、learned-likelihood、GRWRを再生成し、保存済み20 selectorと15 final LightGBMだけを適用する。
- 学習はselector train / final trainに分離したままとし、hidden-testでのselector inferenceとfinal inferenceだけを単一notebookに統合する。
