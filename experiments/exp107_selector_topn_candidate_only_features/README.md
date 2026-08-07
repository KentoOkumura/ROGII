# exp107_selector_topn_candidate_only_features

## 状態

- 状態: completed_train_side_rejected
- Route: MLモデル
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 提出: なし

## 仮説

exp098 の rank-slot features は有用だったが、`sc_ens` / `hyb` のように rank slot へほぼ入らない候補由来の統計や source flag がノイズになっている可能性がある。

上位 n 件に入った候補だけから delta、score、source code、U-space shape、top-n 内 disagreement を作れば、候補集合全体依存のノイズを落としつつ rank-slot signal を残せるかを検証した。

## 検証方針

- exp098 と同じ exp073/exp072 196-feature surface を使う。
- `top1_candidate_only`、`top2_candidate_only`、`top3_candidate_only` を同じ GroupKFold by well で ablation する。
- direct selector / soft average / postprocess replacement は行わない。
- 比較対象は exp073 raw anchor、exp077 policy、exp092 best、exp098 full rank-slot、exp105 compact。

## 所見

Kaggle train v1 の best は `top2_candidate_only` / `lgb2` RMSE 9.437602823。

exp105 compact はわずかに上回ったが、exp098 full rank-slot lgb1 9.358151052 と exp092 lgb1 9.322479896 には届かない。top-n candidate-only の追加列 pruning は rejected とし、提出しない。

## 主要ファイル

- 学習 notebook: `exp107_selector_topn_candidate_only_features_train.ipynb`
- 推論 notebook: `exp107_selector_topn_candidate_only_features_inference.ipynb`
- 実装: `selector_topn_candidate_only_features.py`
- 設定: `config.yaml`
- Kaggle output: `kaggle/output/train_v1`
