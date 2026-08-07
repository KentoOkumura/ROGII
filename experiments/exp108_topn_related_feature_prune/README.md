# exp108_topn_related_feature_prune

## 状態

- 状態: completed_train_side_rejected
- Route: MLモデル
- 親実験: `exp098_selector_rank_slot_features_on_exp073`
- base surface: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 提出: なし (OOF 悪化のため inference / submit しない)

## 仮説

exp098 の full 260 feature surface には、rank slot にほぼ入らない `sc_ens` / `hyb` family、source one-hot、全候補統計、広い pairwise disagreement が残っている。

追加 rank-slot 列だけでなく、既存 196 features 側も含めて top-n selector と関係が薄い candidate family / rank-slot signal を静的に落とすと、exp098 の有効な signal を残しつつ noise を減らせるかを検証する。

## 検証方針

- exp098 と同じ exp073/exp072 196-feature surface を使う。
- exp098 full 260 は config に control として残すが、GPU 節約のため学習しない。
- top-n は既存 exp098 の selector 分布と特徴量重要度から top3 に固定する。
- active variant は `top3_related_pruned_260` のみとする。
- row-wise dynamic masking、direct selector、soft average、candidate TVT replacement は行わない。
- inference は train OOF、worst-well、bucket、feature importance、path continuity を確認するまで対象外。

## 主要ファイル

- 学習 notebook: `exp108_topn_related_feature_prune_train.ipynb`
- 推論 notebook: `exp108_topn_related_feature_prune_inference.ipynb`
- 実装: `topn_related_feature_prune.py`
- 設定: `config.yaml`

## 所見

2026-06-22 に Kaggle train v1 を完了した。active variant は `top3_related_pruned_260` のみで、使用 features は 195。

pooled OOF best は `lgb2` 9.479370656、`lgb_mean` は 9.529005954。exp073 raw anchor 9.526374749 より best single では -0.047004 改善したが、exp098 best 9.358151052 より +0.121220、exp105 best 9.441103161 より +0.038267、exp077 policy 9.470514801 より +0.008856、exp092 best 9.322479896 より +0.156891 悪い。

top3 関連 feature への静的 pruning は、full 260 surface から有効な context / disagreement / candidate-family signal を落としすぎるため rejected。inference / submit は行わない。

top3 固定の根拠:

- exp098 の rank3 source distribution は `pf_ancc` 41.26%、`beam_mean` 52.26%、`likpf_mean` 6.48% で、rank1/rank2 と異なる candidate family 情報を持つ。
- exp098 の特徴量重要度では `rank3_u_curvature`、`rank3_u_slope`、`rank3_u_resid_mad`、`rank3_candidate_minus_last_anchor` が上位に入っている。
- `sc_ens` / `hyb` は rank3 でもほぼ選ばれないため、top4/top5 へ広げる根拠は薄い。
