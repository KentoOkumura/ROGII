# exp197_cnn_pf_likelihood_probe 結果

## 仮説

PF の point-GR likelihood を local CNN/SDF likelihood に置き換える余地があるかを、固定 PF/Beam 候補上の scorer として検証する。real GR が shuffled/no-GR control を上回り、かつ exp099 multiobs / exp111 / likPF baseline に対して candidate AUC または topK coverage の改善を示す場合だけ後続に進む。

## 設定

- 親: `cnn_pf_likelihood_probe` backlog / cache parent `exp099_pf_multi_observation_likelihood_probe`
- 検証: GroupKFold by well fold0, train pseudo-tail fixed candidates
- メトリック: candidate AUC, topK coverage, weighted RMSE, ESS, worst-well, negative control margin
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| Kaggle kernel | `kentookumura/exp197-cnn-pf-likelihood-probe-train` v1 COMPLETE |
| CV / 主指標 | candidate AUC |
| real_gr learned_prob AUC | 0.9086916392 |
| shuffled_gr learned_prob AUC | 0.9027273274 |
| no_gr learned_prob AUC | 0.9053030435 |
| real - shuffled AUC | +0.0059643118 |
| real - no_gr AUC | +0.0033885957 |
| exp111 learned probability AUC | 0.9158250218 |
| exp099 multiobs score AUC | 0.6121555671 |
| point-GR likelihood AUC | 0.5690632201 |
| real_gr learned_prob top1 RMSE / MAE / within10 | 11.301053 / 6.735788 / 0.784917 |
| likPF single top1 RMSE / MAE / within10 | 11.293248 / 6.764031 / 0.785750 |
| learned_error top1 RMSE / MAE / within10 | 11.252965 / 6.708288 / 0.784500 |
| learned_prob top2 oracle RMSE / within10 | 9.420265 / 0.843750 |
| learned_prob top3 oracle RMSE / within10 | 8.117179 / 0.892167 |
| learned_prob top5 oracle RMSE / within10 | 7.774709 / 0.903417 |
| Public LB | - |
| Private LB | - |

Notebook decision は `weak_real_gr_signal_needs_guarded_followup`。`learned_top1_delta_rmse_vs_likpf` は +0.007804 で、learned_prob top1 は likPF single よりわずかに悪い。

## 再現性

- deterministic anchor: false
- seed policy: fixed global seed + SHA256 stable row subsample / shuffled-GR roll
- kernel version: `kentookumura/exp197-cnn-pf-likelihood-probe-train` v1
- feature schema SHA: `e1fe9bfc900e1f4bf8475fa8403eaf7cbfc6f4f68c1b7b1a3bdd9dd04f5af6c3`
- candidate index SHA: gz `e48532343a61209be23e6028540e15ae3c19bc1751d3b2ff169311441fcb32fd`, decompressed `d78918f24934c394c40e883cad2207e1c66b629f9ec1bb3146cda8f95fae4e6b`
- model manifest SHA: `8e2b0b3b97954b1bd95e6514e4ae750702fa1d7f46ad44efd34b3e07b4ed1188`
- prediction SHA: OOF probability `29795eeb9d5771097b611ad2a66a19ee58c172bf05a3b323dc6862bdefb88e59`, expected error `cf5a2a6820498acd768a1267dbc82080ec35ea9a7c673d161871246001f83634`
- summary JSON SHA: `b379301a667d793e9382fc0c4c4e946c134e500a3e2dc78b6f06f0c43d343ef0`
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

candidate likelihood としては強い scorer が学習できているが、real GR の寄与は弱い。real_gr は shuffled_gr に +0.006 AUC、no_gr に +0.003 AUC しか上乗せしておらず、raw local GR そのものより candidate scalar / row context / 既存候補構造を拾っている可能性が高い。

top1 の実用面でも、learned_prob は likPF single に RMSE で +0.0078 悪く、within10 も 0.00083 低い。learned_error top1 は RMSE/MAE では少し良いが、within10 は likPF より低い。したがって PF weight replacement、raw-test feature generation、submit へ進める根拠はない。

exp111 learned probability は AUC 0.9158 で exp197 real_gr learned_prob より高い。exp197 は discussion 699853 の local CNN/SDF likelihood idea を train-side diagnostic として検証し、少なくとも現設定では「GR window CNN を直接 PF likelihood に入れる」優先度を下げる結果。

## 次

追加でやるなら、candidate scalar / row context を強く制限した scalar-only / GR-only ablation で real GR の純粋寄与を再確認する。ただし現時点では高優先ではない。PF/Beam 本線では、exp111 系 learned likelihood や既存 likPF の安定化を優先する。
