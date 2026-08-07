# exp225_state_known_tvt_self_gr_hmm_emission 結果

## 状態

Kaggle train v1 完了。state-known self-GR emission は exp072 `likpf_mean` より大きく悪化したため、不採用。raw-test regeneration、inference、submit には進めない。

## 仮説

exp223 の self-GR motif boost は train-side で exp072 `likpf_mean` を改善したが、exp209 HMM/likPF blend には届かず、最大悪化 well が大きかった。今回の exp225 では self-GR を候補値や replacement にせず、known-prefix の `TVT_input -> GR` 曲線が定義できる candidate TVT state だけに弱い emission boost として足した。

## 設定

- 親: `exp223_joint_typewell_self_gr_hmm_likelihood_probe` / `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `ensemble`
- 検証: train-side no-training HMM emission readout
- active variants: 1 (`hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100`)
- LightGBM booster: 0
- control retraining: なし
- inference / submit: なし
- Kaggle kernel: `kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train`
- URL: https://www.kaggle.com/code/kentookumura/exp225-state-known-tvt-self-gr-hmm-emission-train
- version: 1
- status: `KernelWorkerStatus.COMPLETE`

## 結果

- rows / wells: 3,783,989 / 773
- elapsed: 17,310.949 sec (約 4h48m31s)
- best overall: exp072 `likpf_mean`
- exp072 `likpf_mean` RMSE: 11.594897668
- state-known self-GR HMM RMSE: 14.212954500
- delta vs exp072 `likpf_mean`: RMSE +2.618056832、MAE +1.223108278、within10 -0.046363771
- state-known self-GR HMM bias: -5.651001931

Distance bucket は近傍だけ改善し、それ以降は悪化した。`000_050` は -0.235925、`050_100` は -0.316982 だが、`100_250` +0.030487、`250_500` +0.311389、`500_1000` +0.883882、`1000_plus` +2.931795 と longtail で大きく悪化した。

Hidden-like も悪化した。verification_like_spatial は RMSE 16.581602、delta +2.937794。verification_like_typewell_purged は RMSE 16.348911、delta +2.842109。

By-well は 379 wells 改善、394 wells 悪化で、median delta は +0.032908、mean delta は +1.275998。最大悪化は `2fd68f7b` の +49.423573 RMSE、次点も `8d5d46d7` +49.278663、`b19b0395` +48.319029 と大きい。最大改善は `7987f2f2` の -42.177293 RMSE。

HMM posterior は滑らかで、step delta は `abs_step_delta_mean=0.009788`、p99 0.061、5.0 超過率 0.0。ただし滑らかさは global / hidden-like の悪化を相殺しなかった。self-GR state-valid rate は mean 0.529194 / median 0.513089 で、state-valid rate 上位 bucket だけは改善するが、全体の誤誘導を抑えられていない。

## 判定

この新案の「candidate state ごとに known TVT 範囲内か判定し、範囲外は neutral」という実装自体は成立した。しかし train-side の結果は exp072 `likpf_mean` から大幅に悪化し、hidden-like と longtail も悪化した。exp223 の motif boost より worst-well regression も小さくならなかった。

したがって `state_known_tvt_self_gr_hmm_emission` は完了・不採用。追加 alpha / sigma grid、raw-test regeneration、inference、submit は行わない。self-GR を HMM emission の直接補正として使う方向は優先度を下げ、使う場合は ML / selector の confidence feature や regression guard 用 readout に限定する。
