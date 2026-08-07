# exp134_self_gr_multiscale_longtail_gate 結果

## 状態

Kaggle train v1 完了。直接 gate / inference port / submission は棄却。

## 仮説

exp090 の self-GR multiscale signal を、単独予測ではなく high-drift / PF-dense disagreement gate の補助 confidence として使う。

## 評価方針

LightGBM は学習しない。exp072 full replay cache と raw train wells から self-GR signal を再生成し、`likpf_mean` baseline と `tvt_dense` low-frequency gate variants を posthoc に比較する。評価は overall RMSE、common worst、tail bucket、near rows、PF-dense disagreement、by-well regression を見る。

## 結果

3,783,989 rows / 773 wells で posthoc audit を完了した。LightGBM 学習はしていない。

| prediction | RMSE | delta vs `likpf_mean` | gate rate |
| --- | ---: | ---: | ---: |
| `pred_exp092_lgb1` | 9.322480 | -2.272418 | 1.000000 |
| `pred_exp073_lgb_mean` | 9.526375 | -2.068523 | 1.000000 |
| `pred_likpf_mean` | 11.594898 | 0.000000 | 1.000000 |
| `pred_dense_longtail_pf_dense_q4_self_q75` | 15.304252 | +3.709355 | 0.058405 |
| `pred_dense_tail500_pf_dense_q4_self_q75` | 15.567122 | +3.972224 | 0.063317 |
| `pred_dense_longtail_pf_dense_q4_self_q60` | 17.084310 | +5.489412 | 0.092400 |
| `pred_dense_longtail_pf_dense_q4` | 22.105710 | +10.510812 | 0.223399 |
| `pred_tvt_dense` | 23.470396 | +11.875499 | 1.000000 |

`self_gr_q75` 条件は、self-GR なしの dense gate よりは破壊を抑えたが、`likpf_mean` より大幅に悪い。`1000_plus + pf_dense_diff_q4` でも `likpf_mean` RMSE 19.047514 に対し、`self_q75` gate は 28.450639、self 条件なし gate は 44.141818 だった。

common worst 26 wells では `tvt_dense` 単体 RMSE 20.539466、self 条件なし gate 21.537070 と `likpf_mean` 36.823806 より良いが、self-GR 条件を足すと `self_q75` gate は 32.578503 まで戻ってしまう。さらに by-well 最大悪化は `self_q75` gate で +96.835970 RMSE と大きく、worst-well guard を通らない。

Kaggle evidence:

- kernel: `kentookumura/exp134-self-gr-gate-train` v1
- output: `experiments/exp134_self_gr_multiscale_longtail_gate/kaggle/output/train_v1`
- rows / wells: 3,783,989 / 773
- runtime: 3315.821 sec
- input decompressed cache SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- gate predictions decompressed SHA: `132288cd7dfa93d62c3ddae0ba182668d3790ebd451b3c285f72fb10ce8604b0`

## 次

`self_gr_multiscale_longtail_gate` は完了として閉じる。self-GR 条件は direct gate、inference port、submit、単独 follow-up には進めない。

`tvt_dense` は common-worst に headroom があるが、self-GR quality はその選別には不十分だった。dense 系を使うなら、self-GR ではなく segment-level selector / verifier、PF-dense disagreement、path continuity、raw-test-compatible confidence を優先する。
