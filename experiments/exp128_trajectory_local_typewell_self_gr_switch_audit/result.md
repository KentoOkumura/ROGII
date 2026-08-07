# exp128_trajectory_local_typewell_self_gr_switch_audit 結果

## 仮説

typewell GR cost が局所的に悪い窓では、同じ horizontal well の visible prefix にある GR motif への self-match が、PF/Beam / likelihood-PF 候補の局所補正として使える可能性がある。ただし self-GR direct candidate は exp091 で大きく悪化したため、候補生成は cost gap による保守的な switch / blend の監査に限定する。

## 設定

- 親: exp099_pf_multi_observation_likelihood_probe
- 検証: exp099 train-side OOF candidate cache 上の疑似 tail rows
- メトリック: RMSE / MAE / within10 / bucket RMSE / by-well regression / switch rate
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 11.594897672217703 |
| Public LB | - |
| Private LB | - |

### v2 正式結果

| 項目 | 値 |
| --- | --- |
| rows | 3,783,989 |
| wells | 773 |
| best candidate | `likpf_mean` |
| best RMSE | 11.594897672217703 |
| best MAE | 7.067632584311985 |
| best within10 | 0.772807479091509 |
| delta vs `likpf_mean` | 0.0 |
| finite self prior rate | 0.7563457504765474 |
| mean `typewell_cost - self_cost` | -0.7429656386375427 |
| best switch/blend gate | 0.0 |

## 再現性

- deterministic anchor: いいえ。train-side audit であり提出生成はしない。
- seed policy: 新規処理は RNG 不使用の single-process window scan。
- kernel version: `kentookumura/exp128-trajectory-local-switch-train` v2。
- feature content SHA: OOF gzip decompressed SHA `fa274d49641bfbf033817f79971e20a9678499336b449d312601a6066bfa5731`。
- model SHA / manifest SHA: 学習モデルなし。
- prediction SHA: OOF gzip raw SHA `972ac359b8266399c2978822cded25677dd2516ffb584bd79913a1c334c29bc6`。
- submission SHA: 提出なし。
- rerun result: v1 は invalid、v2 を正式結果として採用。

## 解釈

v1 は完了したが、soft blend 候補が `0 * NaN` 伝播で self prior finite subset だけを評価していたため invalid。見かけ best は RMSE 11.552085575 / coverage 0.756345750、baseline `likpf_mean` は RMSE 11.594897672 / coverage 1.0 で比較面が不一致だった。

v2 では soft blend を修正し、全候補を full-row coverage で公平に比較した。結果は best が `likpf_mean` のままで、local switch / blend は gate 0.0、switch 0.0 のため改善なし。`local_cost_gap_typewell_minus_self` の平均は -0.742966 で、self-GR prefix match は typewell cost より弱い。`self_gr_prefix_prior_tvt` は worst-well で数千 ft 規模に壊れるため、直接候補としても hard switch source としても採用しない。

## 次

この実験は completed / rejected / no submit として閉じる。self-GR は trajectory 内の local switch ではなく、使うとしても high-drift / PF-dense disagreement gate の補助 confidence 程度に限定する。
