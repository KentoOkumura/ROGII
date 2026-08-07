# exp230_dtw_typewell_warp_hmm_emission_correction 結果

## 仮説

exp209 exact HMM に constrained DTW/typewell warp の弱い補助 emission を足すと、DTW path を直接候補化せずに GR alignment signal を使える。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- HMM: exp209 と同一 grid / transition / raw GR emission
- DTW variants: `hmm_dtw_a005_s1200`, `hmm_dtw_a010_s1200`
- LightGBM config / folds / boosters: 0 / 0 / 0
- exp072 replay: 再生成なし。保存済み cache を comparison baseline として使用。
- 推論 / 提出: なし

## 結果

- Kaggle train v2 `kentookumura/exp230-dtw-hmm-emission-train`: `COMPLETE`
- rows / wells: 3,783,989 / 773
- elapsed: 36,768.8 sec
- best overall: `exp072_likpf_mean` RMSE 11.594897668
- best DTW-HMM: `hmm_dtw_a005_s1200` RMSE 13.611292323、exp072 `likpf_mean` から +2.016394654
- `hmm_dtw_a010_s1200`: RMSE 16.435494713、exp072 `likpf_mean` から +4.840597045
- hidden-like: `hmm_dtw_a005_s1200` は spatial +1.645723、typewell-purged +1.740356 RMSE 悪化
- distance bucket: `000_050` -0.279896、`050_100` -0.424749、`100_250` -0.224815、`250_500` -0.082857 と近距離は小改善。一方 `500_1000` +0.128648、`1000_plus` +2.300709 と long-tail で悪化
- by-well: `hmm_dtw_a005_s1200` は 409 wells 改善 / 364 wells 悪化、最大悪化は `b19b0395` +47.803293 RMSE
- step delta spikes: `hmm_dtw_a005_s1200` は >5 / >10 / >25 ft がすべて 0

## 解釈

DTW 補助 emission は近距離 bucket を少し改善するが、評価行の大半を占める `1000_plus` と hidden-like subgroup を壊す。alpha 0.10 はさらに悪化し、alpha 0.05 でも exp072 `likpf_mean` に届かない。DTW anchor confidence は直接 HMM emission に足すには局所改善より long-tail regression が大きい。

この結果から、`dtw_typewell_warp_hmm_emission_correction` は完了・不採用。raw-test regeneration、inference、submit、追加 alpha grid は行わない。

## 次

DTW / elastic registration を使う場合は、HMM emission への直接補正ではなく、ML / selector 側の confidence feature、または regression guard readout に限定する。
