# exp223_joint_typewell_self_gr_hmm_likelihood_probe 結果

## 状態

Kaggle train v1 完了。train-side diagnostic としては exp072 likPF に対して positive。ただし exp209 HMM/likPF blend 水準には届かず、worst-well regression が大きいため、raw-test port / submit には進めない。

## 仮説

exp209 exact HMM の typewell GR emission は維持しつつ、同一 horizontal well の visible prefix GR motif から作る self-GR likelihood を弱く足せば、typewell GR だけでは曖昧な row で posterior を少しだけ改善できる可能性がある。

ただし exp091 / exp128 / exp134 で self-GR direct candidate、hard switch、dense gate は不採用なので、今回も self-GR を候補値や replacement としては使わない。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `ensemble`
- 検証: train-side no-training HMM emission readout
- active variants: 2 (`hmm_selfgr_boost_only_a070_c100`, `hmm_selfgr_boost_only_a150_c100`)
- LightGBM booster: 0
- control retraining: なし
- inference / submit: なし
- Kaggle kernel: `kentookumura/exp223-selfgr-hmm-train`
- URL: https://www.kaggle.com/code/kentookumura/exp223-selfgr-hmm-train

## 結果

- rows / wells: 3,783,989 / 773
- elapsed: 39,029.366 sec (約 10h50m29s)
- best: `hmm_selfgr_boost_only_a070_c100`
- RMSE: 11.349950650
- MAE: 6.471271592
- within10: 0.794830006
- exp072 `likpf_mean` baseline RMSE: 11.594897668
- delta vs exp072 `likpf_mean`: RMSE -0.244947018、MAE -0.596360991、within10 +0.022027812
- `hmm_selfgr_boost_only_a150_c100`: RMSE 11.559594481、delta vs exp072 `likpf_mean` -0.035303188

Distance bucket は全 bucket で exp072 `likpf_mean` より改善した。`hmm_selfgr_boost_only_a070_c100` の delta RMSE は、000_050 -0.283719、050_100 -0.412821、100_250 -0.196443、250_500 -0.179257、500_1000 -0.396934、1000_plus -0.247524。

Hidden-like も改善した。verification_like_spatial は RMSE 12.463389、delta -1.180418。verification_like_typewell_purged は RMSE 12.266304、delta -1.240497。

一方で by-well regression は重い。461 wells 改善、312 wells 悪化、median delta -0.549381 だが、最大悪化は `b19b0395` の +46.954683 RMSE。`97cd5bf9` +36.021001、`8a3da6d1` +35.692945 など大きい悪化 well がある。

HMM std calibration は概ね std が高いほど error が大きくなるが、low-std bin でも RMSE 9.365031 があり、過信 guard は必要。step delta は exp072 より滑らかで、`abs_step_delta_mean` は 0.010138、p99 は 0.061。

## 判定方針

global RMSE が小改善でも、worst-well、near-row、hidden-like stress、self-GR disagreement bucket が弱い場合は diagnostic で閉じる。raw-test port / submit には進めない。

## 判定

初回仮説は「self-GR motif likelihood を弱く足すと exp072 likPF より良い HMM posterior になり得る」という点では支持。ただし exp209 の HMM/likPF blend best RMSE 10.269696 には届かず、worst-well regression が大きい。したがって、このまま raw-test-safe regeneration、inference、submit へは進めない。

後続で使うなら、直接候補や replacement ではなく、ML / selector 側の confidence feature、または regression guard 付き診断材料に限定する。`alpha=0.15` は `alpha=0.07` より弱いため、追加 grid を広げる優先度は低い。
