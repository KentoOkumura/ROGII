# exp127_learned_likelihood_features_on_exp092 結果

## 状態

Kaggle train v1 完了。提出なし。

- Kernel: `kentookumura/exp127-train`
- URL: https://www.kaggle.com/code/kentookumura/exp127-train
- Output: `kaggle/output/train_v1`
- Runtime: 7133.981 sec
- 評価行: 757,738 rows / 155 wells

## 仮説

exp112 の learned likelihood feature cache を exp092 系 ML に add-only confidence feature として入れると、candidate path の信頼度や不確実性を LightGBM が吸収し、exp092 shared-row control を上回る可能性がある。

## 評価設計

exp112 feature cache が 155 wells subset のため、exp092 full CV との直接比較ではなく、同じ shared rows 上の `exp092_shared_row_control` と `learned_likelihood_confidence_addonly` を比較する。

## 結果

pooled OOF では `learned_likelihood_confidence_addonly` が shared-row control を全モデルで改善した。

| variant | model | features | RMSE |
| --- | --- | ---: | ---: |
| `exp092_shared_row_control` | `lgb0` | 240 | 10.022150359 |
| `exp092_shared_row_control` | `lgb1` | 240 | 9.865476965 |
| `exp092_shared_row_control` | `lgb2` | 240 | 9.872184867 |
| `exp092_shared_row_control` | `lgb_mean` | 240 | 9.847052694 |
| `learned_likelihood_confidence_addonly` | `lgb0` | 294 | 9.867141369 |
| `learned_likelihood_confidence_addonly` | `lgb1` | 294 | 9.773581370 |
| `learned_likelihood_confidence_addonly` | `lgb2` | 294 | 9.753643527 |
| `learned_likelihood_confidence_addonly` | `lgb_mean` | 294 | 9.727317518 |

差分は `lgb0` -0.155008989、`lgb1` -0.091895595、`lgb2` -0.118541340、`lgb_mean` -0.119735177。distance bucket は 0-50、50-100、100-250、250-500、500-1000、1000+ の全 bucket で改善した。

by-well の `lgb_mean` 差分は平均 -0.042303、中央値 -0.048879、p75 +0.199447、最悪悪化 +1.071012、最大改善 -2.155674。最悪悪化は `aed44918` の 9.860902 -> 10.931914、最大改善は `1b1eba53` の 60.608173 -> 58.452499。

上位 learned likelihood feature importance は `ll_candidate_tvt_likpf_mean_minus_last_known_tvt`、`ll_learned_prob_beam_mean`、`ll_learned_pred_abs_error_beam_mean`、`ll_learned_prob_weighted_tvt_minus_last_known_tvt`、`ll_candidate_tvt_beam_mean_minus_last_known_tvt`。

## 解釈

shared rows 上では、exp112 learned likelihood confidence feature を exp092 系 LightGBM に add-only で渡す仮説は支持された。特に `lgb_mean` の -0.119735 RMSE と全 distance bucket 改善は、hard gate ではなく ML feature として使う方向を支持する。

一方で、これは exp112 cache が存在する 155 wells subset の評価であり、exp092 full CV や hidden-like split の直接証拠ではない。by-well の最悪悪化 +1.071012 も残るため、このまま inference port / submit はしない。次に使う場合は exp115 hidden-like stress、raw-test/full-train feature parity、worst-well guard を通し、`segment_level_dense_candidate_verifier` や他の confidence feature に入力として再利用する。
