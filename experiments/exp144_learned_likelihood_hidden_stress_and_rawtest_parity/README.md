# exp144_learned_likelihood_hidden_stress_and_rawtest_parity

## 状態

Kaggle train v1 完了。提出なし。

## 仮説

exp127 では exp112 learned likelihood confidence features を exp092 系 LightGBM に add-only で渡すと、shared rows 上で control を改善した。一方、hidden-like split と raw-test/full-train parity が未確認なので、後続の confidence feature や segment verifier に渡す前に stress readout と parity blocker を明示する。

## 検証方針

新規学習はしない。次の保存済み生成物を読む。

- exp127 row-level OOF predictions
- exp127 feature schema / summary
- exp112 learned likelihood ML feature cache
- exp115 hidden-like fold assignments / well metadata

`exp092_shared_row_control` と `learned_likelihood_confidence_addonly` を、all shared rows、`verification_like_spatial`、`verification_like_typewell_purged` で比較する。

## 所見

exp115 hidden-like stress でも `learned_likelihood_confidence_addonly` は `exp092_shared_row_control` を改善した。

- all shared rows: `lgb_mean` 9.847053 -> 9.727318、delta -0.119735。
- `verification_like_spatial`: `lgb_mean` 13.037491 -> 12.760311、delta -0.277180。
- `verification_like_typewell_purged`: `lgb_mean` 13.082838 -> 12.787921、delta -0.294917。

near `000_050` と `1000_plus` longtail も hidden-like split で改善した。一方、hidden-like well regression は最大 +1.071000 RMSE 残り、raw-test parity checklist は full-train coverage 155/773 wells と raw-test feature regeneration missing で fail した。

## 生成物

- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_overall_metrics.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_bucket_metrics.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_by_well.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_overall_delta.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_bucket_delta.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_worst_well_delta.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_rawtest_parity_checklist.csv`
- `exp144_learned_likelihood_hidden_stress_and_rawtest_parity_summary.json`

## 判断

hidden-like stress は支持。ただし raw-test/full-train parity が未充足のため提出候補にはしない。exp127 learned likelihood feature family は、raw-test generator を作るまでは confidence diagnostic / segment verifier 材料に限定する。
