# exp105_compact_rank_slot_features_on_exp098

## 状態

- Route: `ml_model`
- 状態: `completed_train_side_rejected`
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- Kaggle Notebook 実行を正とする。ローカル notebook 実行はしない。

## 仮説

exp098 の rank-slot 特徴量は exp073 / exp077 を改善したが、64 列の追加特徴量には重複列、符号反転ペア、importance 0 に近い列、ほぼ選ばれない `sc_ens` / `hyb` flag が含まれる。

rank1/rank2/rank3 の主要信号だけを残せば、LightGBM の分岐候補を減らしつつ exp098 の有効信号を保てる可能性がある。

## 検証方針

- exp098 と同じ exp073/exp072 196-feature full replay surface、target、GroupKFold by well、LightGBM family を使う。
- rank-slot 候補値は直接 selector / soft average / postprocess replacement として使わない。
- active variant は `compact_rank_slot_features` のみ。
- compact group は 22 列:
  - `rank*_candidate_minus_last_anchor`
  - `rank*_score`
  - `rank*_source_code`
  - `rank*_u_slope`
  - `rank*_u_curvature`
  - `rank*_u_resid_mad`
  - `rank_score_entropy`
  - `rank_score_top1_margin`
  - `rank_slot_u_std`
  - `rank_slot_u_range`
- exp098 `lgb1` 9.358151052、exp098 `lgb_mean` 9.427447987、exp092 `lgb1` 9.322479896 と比較する。

## 所見

Kaggle train v1 は完了。best は `lgb2` 9.441103161 で、exp098 `lgb1` 9.358151052 より悪化した。compact 22-column feature set は提出候補にしない。

## ファイル

- 学習 notebook: `exp105_compact_rank_slot_features_on_exp098_train.ipynb`
- 推論 notebook: `exp105_compact_rank_slot_features_on_exp098_inference.ipynb`
- 実装: `compact_rank_slot_features_on_exp098.py`
- 設定: `config.yaml`
