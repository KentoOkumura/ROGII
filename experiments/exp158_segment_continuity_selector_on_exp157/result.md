# exp158_segment_continuity_selector_on_exp157 結果

## 状態

Kaggle train v1 完了。提出なし。train-side supported だが、inference port / submit はまだしない。

## 仮説

exp157 の row-wise error ranker は dense 候補追加により `likpf_mean` を大きく超えたが、path switch が大きい。exp157 の predicted-error surface を Viterbi で連続化し、短い非 default segment を戻すことで、精度改善を保ちながら hidden-like な path instability を抑えられるかを診断する。

## 設定

- Kernel: `kentookumura/exp158-segment-continuity-selector-on-exp157-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp158-segment-continuity-selector-on-exp157-train`
- output: `experiments/exp158_segment_continuity_selector_on_exp157/kaggle/output/train_v1`
- 親: `exp157_candidate_ranker_feature_enrichment`
- cache: `exp099_pf_multi_observation_likelihood_probe` v2 train feature cache
- dense cache: `exp072_exp063_full_replay_feature_cache`
- rows / wells: 3,783,989 / 773
- runtime: 21,394.101 sec
- default: `likpf_mean`
- switch 候補: `pf_ancc`、`beam_mean`、`sc_ens`、`hyb`、`tvt_dense`、`tvt_densew`、`tvt_dense50`
- 追加学習: なし
- exp157 saved booster inference: 15 boosters
- Viterbi variants: 180

## 結果

best Viterbi は exp157 row-wise をわずかに超えた。RMSE は `likpf_mean` から -0.805734、exp157 row-wise から -0.006590 改善した。within10 も exp157 row-wise から +0.000142 改善した。

| variant | RMSE | MAE | within10 | non-default rate |
| --- | ---: | ---: | ---: | ---: |
| `oracle` | 4.564605 | 2.317166 | 0.960054 | 0.736003 |
| `viterbi_sw050_bias000_jw050_jf025_d150_std999999_md0000_seg001` | 10.789163 | 6.469585 | 0.792647 | 0.616326 |
| `exp157_error_ranker_rowwise` | 10.795753 | 6.476904 | 0.792505 | 0.620559 |
| `likpf_mean_single` | 11.594898 | 7.067633 | 0.772807 | 0.000000 |

best Viterbi の選択分布:

| candidate | rows | rate |
| --- | ---: | ---: |
| `pf_ancc` | 1,475,809 | 0.390014 |
| `likpf_mean` | 1,451,817 | 0.383674 |
| `tvt_dense50` | 316,207 | 0.083564 |
| `tvt_densew` | 281,997 | 0.074524 |
| `beam_mean` | 171,234 | 0.045252 |
| `tvt_dense` | 78,052 | 0.020627 |
| `hyb` | 6,871 | 0.001816 |
| `sc_ens` | 2,002 | 0.000529 |

## Continuity

Viterbi は row-wise の path switch を大きく減らした。

| variant | total path switches | path switches / 1000 rows | max well path switches / 1000 rows |
| --- | ---: | ---: | ---: |
| best Viterbi | 11,767 | 3.109681 | 24.091920 |
| exp157 row-wise | 277,110 | 73.232242 | 357.199056 |
| `likpf_mean_single` | 0 | 0.000000 | 0.000000 |

by-well では exp157 row-wise 比で 428 wells 改善、345 wells 悪化。最大 regression は +1.906477 RMSE、最大 improvement は -3.919611 RMSE。`likpf_mean` 比では 480 wells 改善、293 wells 悪化。

## Guardrail

best Viterbi の worst well は引き続き `86454a6f` で、RMSE 57.836738、within10 0.049598。exp157 row-wise の同 well RMSE 57.967201 よりは少し良いが、絶対値としてはまだ重い。

主要 bucket は `likpf_mean` より広く改善している。

| bucket | best Viterbi RMSE | likpf RMSE | delta |
| --- | ---: | ---: | ---: |
| distance `000_050` | 0.483925 | 1.188878 | -0.704953 |
| distance `050_100` | 1.128152 | 1.925625 | -0.797473 |
| distance `1000_plus` | 11.841614 | 12.704015 | -0.862401 |
| `pf_seed_std_q4` | 10.528585 | 11.071060 | -0.542475 |

exp157 row-wise 比では多くの bucket が微改善だが、distance `500_1000` と eval_len q3 など一部はわずかに悪化した。

## 再現性

- exp099 train feature decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp072 auxiliary source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp157 feature schema SHA: `891226fdf0f82c384e2fcca77f3c7d47b964d5837251ce9594249951d4e5b87c`
- exp157 model manifest SHA: `ab25fbfc0c8b92915bfbd11e62c8ffa6d84eadb3d8abf10e039927e2df7d4fb1`
- metrics SHA: `35828ea61896fae4b0cb463e7391cd62d59dc1abc6df1abd53863225fa172b2c`
- OOF predictions decompressed SHA: `7401d54395939100ca31fa131e74a16992d41936b86f3bc3f6142ed597f452a2`
- best Viterbi prediction SHA: `36f66b1547fbdac2d6bf3b3d8044d89ef0c0be0c4a6bd17e0c4874ec0f790b0f`

## 解釈

exp158 は、exp157 の「dense 候補を入れた selector は効く」という結果を壊さず、row-wise path switch を大きく抑えた。RMSE 改善は exp157 row-wise 比では小さいが、path switch は 277,110 から 11,767 へ大きく減っており、train-side の continuity audit としては supported。

一方で、worst well の絶対 RMSE はまだ非常に大きい。global OOF が良くても、このまま submit する判断はしない。次に進むなら、同じ exp158 内で inference parity を確認し、hidden-like / worst-well guard と raw-test feature availability を見たうえで、submit 候補にするか判断する。

## 次

新しい実験は切らず、必要なら exp158 の中で inference port / raw-test parity audit を行う。提出判断は、worst-well と path continuity の残リスクを確認してからにする。
