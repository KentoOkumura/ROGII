# exp125_confidence_gate_continuity_rawtest_parity 結果

## 仮説

`exp102` / `exp112` の confidence gate は global OOF の小改善だけでは submit 候補にできない。shared surface 上で continuity、worst-well、raw-test parity checklist を確認すれば、直接 gate として進めるか、ML feature / 診断へ戻すかを判断できる。

## 設定

- 親: `exp102_confidence_gated_likpf_fallback_on_exp101`
- 診断親: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- 検証: saved OOF posthoc audit
- メトリック: RMSE、MAE、within10、switch / continuity、worst-well delta
- シード: 42。ただし exp125 自体は RNG なし。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | fair shared best RMSE 11.540333945 (`exp102 gate_error_margin_sr050_d020_std000020`) |
| Public LB | - |
| Private LB | - |

## 再現性

- deterministic anchor: いいえ
- seed policy: `no_new_rng_posthoc_saved_oof_audit`
- kernel version: `kentookumura/exp125-cg-continuity-train` v1
- feature content SHA: exp102 OOF decompressed `469e9fa137...`、exp112 OOF decompressed `e3df222a...`
- model SHA / manifest SHA: exp101 manifest / schema は Kaggle source が追加できず `missing_required`
- prediction SHA: fair prediction content `1af3ae4980362fa246a5a61ac139ab74790bab957c7f2b8a4034a903141147e0`
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

Kaggle train v1 は完了した。fair shared surface は 155 wells / 757,738 rows x 6 variants。best は `exp102` の `gate_error_margin_sr050_d020_std000020` で RMSE 11.540333945、`likpf_mean` から -0.064076236 改善したが、within10 は 0.784311992 -> 0.782674222 へ -0.001637769 悪化した。

`exp112` の `gate_expected_error_m2p0_d20p0` は RMSE 11.573266305、`likpf_mean` から -0.031143876 改善し、within10 は +0.000752239 改善した。ただし direct candidate としては raw-test parity と continuity が足りない。

guardrail は `required_parity_missing_count=2`、`continuity_fail_variant_count=3`、`well_regression_fail_rows=14`、最大 well regression +12.461017。optional dense/high-drift gate prediction は存在しなかった。結論として、exp102/112 gate はこのまま inference port / submit しない。保存済み confidence signal は ML add-only feature または segment selector 診断へ下げる。

## 次

1. `confidence_gate_continuity_rawtest_parity` backlog は完了として閉じる。
2. direct gate ではなく、既存の `learned_likelihood_gate_rawtest_parity_or_ml_feature` / `exp127_learned_likelihood_features_on_exp092` と `segment_viterbi_candidate_selector_on_exp101` 側へ知見を渡す。
3. dense/high-drift gate は、専用 artifact ができるまで exp125 の比較対象にはならない。
