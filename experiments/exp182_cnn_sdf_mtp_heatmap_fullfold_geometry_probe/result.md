# exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe 結果

## 仮説

exp179 の 1 fold smoke で positive だった 5ch heatmap CNN/SDF/MTP が、fold 横断でも real GR signal を使えているか確認する。さらに geometry channel と larger window が topK coverage、worst-well、distance bucket を改善するかを見る。

## 設定

- 親: `exp179_cnn_sdf_mtp_heatmap_probe`
- 検証: 5-fold GroupKFold heatmap MTP geometry diagnostic
- メトリック: top3 within10 center coverage
- シード: 42
- Runtime: Kaggle T4 GPU 想定

## 結果

Kaggle train v1 は完了した。

| メトリック | 値 |
| --- | --- |
| top3 within10 (`base_real_w128_b64_fullfold`) | 0.500000 |
| top3 within10 (`base_shuffled_w128_b64_fullfold`) | 0.218536314914 |
| top3 within10 (`base_no_gr_w128_b64_fullfold`) | 0.071428571429 |
| top3 within10 (`geometry_real_w128_b64_fullfold`) | 0.487710219922 |
| top3 within10 (`geometry_shuffled_w128_b64_fold01`) | 0.206682027650 |
| top3 within10 (`geometry_real_w256_b96_fold01`) | 0.417511520737 |
| top10 within10 (`base_real_w128_b64_fullfold`) | 0.808907780447 |
| top10 oracle RMSE (`base_real_w128_b64_fullfold`) | 13.296284182219 |
| real - shuffled top3 margin (`base`, full-fold) | +0.281463685086 |
| real - no-GR top3 margin (`base`, full-fold) | +0.428571428571 |
| geometry - base top3 margin (`real`, full-fold) | -0.012289780078 |
| worst-well top3 (`base_real_w128_b64_fullfold`) | 0.0 |
| Public LB | - |
| Private LB | - |

| run spec | folds | valid samples | top1 | top3 | top5 | top10 | top10 oracle RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_real_w128_b64_fullfold` | 5 | 10,822 | 0.241545 | 0.500000 | 0.645444 | 0.808908 | 13.296284 |
| `base_shuffled_w128_b64_fullfold` | 5 | 10,822 | 0.079745 | 0.218536 | 0.328775 | 0.545001 | 17.637821 |
| `base_no_gr_w128_b64_fullfold` | 5 | 10,822 | 0.071429 | 0.071429 | 0.071429 | 0.071429 | 134.767278 |
| `geometry_real_w128_b64_fullfold` | 5 | 10,822 | 0.231843 | 0.487710 | 0.640824 | 0.809647 | 11.995428 |
| `geometry_shuffled_w128_b64_fold01` | 2 | 4,340 | 0.072350 | 0.206682 | 0.320968 | 0.549539 | 17.013071 |
| `geometry_real_w256_b96_fold01` | 2 | 4,340 | 0.193088 | 0.417512 | 0.563594 | 0.716129 | 18.643568 |

## 再現性

- deterministic anchor: false。GPU train-side diagnostic として扱う。
- seed policy: fixed global seed + run spec / fold / well keyed SHA256
- kernel version: `kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train` v1
- runtime: Tesla T4 / PyTorch 2.10.0+cu128 / internet disabled
- output archive: `experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/kaggle/output/train_v1`
- sample index decompressed SHA: `f6ec40a1b89e70224127c355a6be31a56e5f38d02f0019505f9a9d45ba0b7d09`
- feature schema SHA: `0216158ba90193566e2a148ab495bb0d8c5bd4b16e1425efcfc8bde8d2013b2d`
- run spec manifest SHA: `4e2a207174f6828c110efa8b503611a3478165be31d5fa2653d7b56186a76686`
- model manifest SHA: `7950bf80c4618198277a973ea82ea21c1c56a4cfc3a846949cc9f9af3eb404c3`
- validation predictions decompressed SHA: `d50f1ee515da7a68f142cae3d918902e111f1384d57dfa2d882093ad560945fa`
- metrics SHA: `6224ff55edcfb4134eae3a706bd689c73aae5a35a90b27b26d704bd780311335`
- summary SHA: `4efeb705601bfd5cf6102e5e63a414c0f1f4402344c541350b6ab95f0f5e6c13`
- submission SHA: なし

## 解釈

`base_real_w128_b64_fullfold` は top3 within10 0.500000 で、`base_shuffled_w128_b64_fullfold` 0.218536 と `base_no_gr_w128_b64_fullfold` 0.071429 を大きく上回った。exp179 の 1 fold smoke だけでなく、773 wells full-fold でも real GR signal は支持される。

ただし geometry channel は primary metric では悪化した。`geometry_real_w128_b64_fullfold` は top10 oracle RMSE 11.995428 と base より良いが、top3 coverage は 0.487710 で base より -0.012290。`geometry_real_w256_b96_fold01` も top3 0.417512 まで落ち、larger window を広げる根拠にはならない。

distance bucket では base real は shuffled より概ね強いが、well metrics では worst-well top3 0.0 が残る。したがって full-length inference、direct TVT replacement、softmax weighted average、PF weight replacement、submission には進めない。この結果は path feature / candidate verifier / confidence feature の材料として扱う。

## 次

`cnn_sdf_mtp_heatmap_fullfold_geometry_probe` は完了。後続は direct replacement ではなく、`base_real_w128_b64_fullfold` の topK path、logit margin、entropy、topK spread、path continuity、shuffled/no-GR gap を exp157/158 系 candidate selector または exp148 系 ML feature へ add-only する診断に限定する。
