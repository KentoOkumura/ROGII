# exp219_ml_tvt_typewell_gr_mismatch_error_detector_on_exp148

## 概要

exp148 の ML OOF 予測 `pred_tvt` を typewell TVT 軸上の仮位置として使い、その周辺 offset で horizontal GR と typewell GR の局所 window 類似度を計算する。目的は TVT を直接補正することではなく、exp148 の high-error row を検出する confidence / error-detector feature として成立するかを no-training readout で確認すること。

## Route

- route: `ml_model`
- status: `completed_readout_rejected_no_addonly_no_submit`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 実行内容: no-training OOF readout

## 状態

Kaggle CPU train v1 完了。LightGBM 学習、inference、submit には進めない判断。

## 仮説

exp148 の ML 予測が正しい row では、`pred_tvt` 上の typewell GR window と horizontal GR window が比較的よく一致する。一方で high-error row では、offset 0 の score 低下、別 offset の優位、entropy 上昇、decoy gap 低下、raw-vs-denoised score gap などが error detector として働く可能性がある。

## 検証方針

exp148 train v1 `lgb_mean` OOF prediction を読み、`pred_tvt + [-50,-25,-10,0,10,25,50]` ft の typewell GR window を horizontal GR window と比較する。出力 feature は `score_at_ml`、`best_offset`、`best_score`、`score_gap`、`entropy`、`decoy_gap`、`derivative_ncc`、`raw_vs_denoised_score_gap`、`local_z_mse`、candidate disagreement interaction など。

評価は `abs_error_gt10` AUC、high-mismatch bucket の error lift、distance bucket、worst-well、exp115 hidden-like subgroup、診断用の小さい offset correction で見る。AUC 0.65 目安と high-mismatch error lift が出る場合だけ、同じ exp219 内で exp148/exp193 add-only LightGBM に進める。

## 注意

`target_tvt`、`abs_error`、`abs_error_gt*` は readout label であり、feature source には使わない。`best_offset` は hard correction、direct replacement、row-wise switch、blend、PF weight replacement に使わない。初期実装では inference / submit は行わない。

## 所見

`mlgr_mismatch_signal` の `abs_error_gt10` AUC は 0.573943 で採用目安 0.65 に届かない。high-mismatch q90 bucket は error_gt_lift 1.632373 で誤差濃縮はあるが、単独 detector としては弱い。diagnostic correction も base exp148 を更新しないため、add-only LightGBM / inference / submit は行わない。
