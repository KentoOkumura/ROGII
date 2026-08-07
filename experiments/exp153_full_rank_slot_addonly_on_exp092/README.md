# exp153_full_rank_slot_addonly_on_exp092

## Status

実装済み、未実行。Colab high-memory 前提の train runner を追加済み。Kaggle/Colab train 結果を確認するまで inference / submit は行わない。

## Hypothesis

exp098 の full rank-slot features は exp073 上では有効だったが、exp092 には届かなかった。exp139 の small subset add-only は exp092 best を更新できず、exp147 の replacement-only は明確に悪化した。

この実験では、exp092 の U-projection correction / disagreement surface は保持し、exp098 と同じ target-free rank-slot feature groups を full add-only で足す。small subset では落ちた列、特に source flag、rank1-3 の U-projection residual/correction、rank-to-rank delta/disagreement を LightGBM に全て渡した場合に非重複ゲインが残るかを確認する。

## Validation Strategy

exp072 deterministic full replay train feature cache、exp092 U-projection settings、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM family を固定する。追加する rank-slot features は exp098 と同じ target-free score で作り、候補 TVT path の direct selector / soft average / blend / postprocess replacement は行わない。

比較対象は exp092 `lgb1` 9.322479896 / Public LB 8.350、exp098 `lgb1` 9.358151052 / Public LB 8.441、exp139 `lgb1` 9.324907641、exp147 best `lgb2` 9.397013393。

## Scope

- Route: `ml_model`
- Parent: `exp092_u_projection_correction_disagreement_fullrun`
- Rank-slot source parent: `exp098_selector_rank_slot_features_on_exp073`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Variant: `u_projection_full_rank_slot_addonly`
- Rank-slot groups: `rank_slot_delta`, `rank_slot_identity_score`, `rank_slot_u_projection`, `rank_slot_u_disagreement`
- Models: `lgb0` / `lgb1` / `lgb2` and `lgb_mean`
- Runtime: Colab high-memory runner preferred, with large cache copied to `/content`

## Expected Outputs

- `exp153_full_rank_slot_addonly_on_exp092_metrics.csv`
- `exp153_full_rank_slot_addonly_on_exp092_by_well.csv`
- `exp153_full_rank_slot_addonly_on_exp092_bucket_metrics.csv`
- `exp153_full_rank_slot_addonly_on_exp092_projection_feature_summary.csv`
- `exp153_full_rank_slot_addonly_on_exp092_rank_slot_feature_summary.csv`
- `exp153_full_rank_slot_addonly_on_exp092_feature_importance.csv`
- `exp153_full_rank_slot_addonly_on_exp092_feature_importance_mean.csv`
- `exp153_full_rank_slot_addonly_on_exp092_feature_importance_mean_top.png`
- `exp153_full_rank_slot_addonly_on_exp092_predictions.csv.gz`
- `exp153_full_rank_slot_addonly_on_exp092_feature_schema.csv`
- `exp153_full_rank_slot_addonly_on_exp092_lgb_models/manifest.json`
- `exp153_full_rank_slot_addonly_on_exp092_summary.json`

## Findings

未実行。現時点の所見は、exp139 small add-only と exp147 replacement-only が exp092 best を更新できなかったため、full rank-slot union の検証は「削った列が効いていたか」と「exp092 上では rank-slot 情報が重複ノイズになるか」を切り分ける実験として扱うこと。

## Decision Gate

採用候補にする最低条件は、exp092 best `lgb1` を pooled OOF で更新し、by-well worst regression、near-row bucket、path continuity、feature importance、Colab runtime log、raw-test feature parity に致命的な悪化がないこと。条件を満たさない場合は train-side rejected として閉じる。
