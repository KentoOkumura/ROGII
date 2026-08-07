# exp123_cross_test_prefix_label_context_audit 結果

## 仮説

同じ pseudo test batch 内の他 well の visible `TVT_input` prefix label から batch-level residual bias / slope / scale が読めるなら、target well tail の prefix-only baseline を診断的に補正できる可能性がある。

## 設定

- 親: `exp037_test_time_prefix_online_training_audit`
- 検証: `GroupKFold` pseudo test batch。target well scoring rows は `TVT_input` NaN tail。
- メトリック: RMSE
- シード: 42
- 候補: `hold_prefix_control`、`self_linear_prefix_control`、`cross_batch_bias_hold`、`cross_batch_slope_hold`、`cross_batch_scale_slope_hold`、`cross_batch_bias_scale_hold`

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 15.909852870734554 |
| Public LB | - |
| Private LB | - |

## 候補別結果

| 候補 | RMSE | MAE | bias | within10 |
| --- | ---: | ---: | ---: | ---: |
| `hold_prefix_control` | 15.909852871 | 11.196479702 | -1.595987179 | 0.578628532 |
| `cross_batch_bias_scale_hold` | 15.917976341 | 11.200670927 | -1.657211631 | 0.578615054 |
| `cross_batch_bias_hold` | 15.920967640 | 11.202312018 | -1.678996308 | 0.578770446 |
| `cross_batch_scale_slope_hold` | 20.375654980 | 15.635932911 | 11.703680824 | 0.409888084 |
| `cross_batch_slope_hold` | 24.204548712 | 19.332532898 | 16.709936772 | 0.310036842 |
| `self_linear_prefix_control` | 1404.728336097 | 1196.201852951 | 1196.201852951 | 0.000000000 |

Fold selection は全 fold で `hold_prefix_control` を選択し、selection RMSE は 15.909852871。

## 再現性

- deterministic anchor: いいえ
- seed policy: sorted file order + deterministic `GroupKFold`
- kernel version: `kentookumura/exp123-cross-test-prefix-label-audit-train` v1
- feature content SHA: no feature cache
- model SHA / manifest SHA: no model
- prediction SHA: row-level prediction は保存しない
- submission SHA: no submission
- rerun result: なし。診断実験として Kaggle train v1 を正とする

## 解釈

他 well の visible `TVT_input` prefix label から作る batch-level bias / slope / scale 補正は、hold baseline を超えなかった。near 51-250 bucket では `cross_batch_bias_scale_hold` が 4.183832 で hold 4.184865 をごくわずかに上回るが、全体、longtail、fold selection では支持されない。

rules risk があるだけでなく、OOF 診断上も改善根拠が弱い。推論化、提出、ML feature 化には進めない。

## 次

`cross_test_prefix_label_context_audit` は完了として閉じる。same-batch 文脈は target-free covariate context / high-drift confidence feature 側を優先する。
