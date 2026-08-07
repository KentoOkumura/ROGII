# exp156_test_batch_covariate_context_audit 結果

## 仮説

test batch 内で同時に見える target-free covariate context から、exp148 `lgb_mean` を信用しにくい high-drift / high-disagreement regime を識別できる可能性がある。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 検証: train-side pseudo-tail posthoc audit
- rows / wells: 3,783,989 / 773
- base: exp148 `lgb_mean`
- fallback candidates: exp073 `lgb_mean`、`likpf_mean`、`tvt_densew`、`tvt_dense50`
- LightGBM training: なし
- inference / submit: なし
- Kaggle kernel: `kentookumura/exp156-test-batch-context-audit-train` v1
- output: `experiments/exp156_test_batch_covariate_context_audit/kaggle/output/train_v1/`

## 結果

| variant | RMSE | MAE | within10 | delta vs exp148 | gate rate | gate wells | max well regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_exp148_lgb_mean` | 8.501281182 | 5.335650921 | 0.856332035 | 0.000000000 | - | - | 0.000000 |
| `context_densew_tail1500_q85_min4_clip10_a025` | 8.502466 | 5.355754 | 0.855262 | +0.001185 | 0.050990 | 138 | +2.321141 |
| `context_exp073_tail500_q70_min3_clip10_a025` | 8.525209 | 5.374387 | 0.854691 | +0.023928 | 0.212327 | 450 | +1.832102 |
| `context_exp073_tail1000_q80_min3_clip15_a035` | 8.549654 | 5.382093 | 0.854872 | +0.048372 | 0.152471 | 358 | +3.651151 |
| `context_likpf_tail1000_q80_min3_clip12_a025` | 8.564378 | 5.347694 | 0.857141 | +0.063096 | 0.152471 | 358 | +2.444652 |

Best non-oracle は `base_exp148_lgb_mean` のまま。context gate は全体 RMSE を改善しなかった。

## 主要 readout

- `context_risk_q4` は base RMSE 10.995253 と高リスク bucket になったが、全 context gate が悪化した。best でも `context_densew` +0.004585、exp073/likPF fallback は +0.132260 以上悪化。
- near `000_050` / `050_100` は gate が発火せず delta 0.0 で守れている。
- `1000_plus + pf_dense_diff_q4` は `context_densew` が -0.006011 とごく小さく改善したが、global 採用根拠には弱い。
- common PF+ML worst26 は `context_densew` が -0.399641、PF worst50 は -0.304031 改善した。一方で global RMSE +0.001185、within10 -0.001070、最大 well regression +2.321141 が残る。
- oracle all candidates は RMSE 3.342165 と headroom は大きいが、target 依存なので採用判断には使わない。
- raw-test parity checklist は gate 条件 target-free / required columns present / no LightGBM training が pass。

## 再現性

- deterministic anchor: いいえ。train-side posthoc diagnostic。
- seed policy: 新規乱数なし。
- exp148 OOF decompressed SHA: `ec28d89641b74c67482aff7a1ebc925db536716f1a024467ae0339dd2326e14d`
- exp073 OOF decompressed SHA: `fd6c68050058c40b4960f3ff2af9905bfcb1c12d540c71e331d0aa85ca9756a4`
- exp072 feature cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- feature schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- submission SHA: 対象外

## 判断

`test_batch_covariate_context_audit` は train-side rejected。inference port / submit は行わない。batch covariate context は risk bucket としては外れやすい領域を捉えたが、fallback 先の選択精度が足りず、exp148 を超える後処理にはならなかった。

次に dense / candidate headroom を使う場合は、batch-level context だけではなく、exp154 の segment-level dense verifier のような candidate path continuity と hidden-like stress を優先する。
