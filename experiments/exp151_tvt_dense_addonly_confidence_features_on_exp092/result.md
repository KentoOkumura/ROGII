# exp151_tvt_dense_addonly_confidence_features_on_exp092 結果

## 仮説

exp092 の U-projection surface に `tvt_dense` family の target-free confidence 特徴を add-only で入れると、target 変更や dense hard switch なしに、exp092 が外れやすい high-disagreement / longtail regime を補助的に表現できる可能性がある。

## 実装

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- route: `ml_model`
- 追加特徴量:
  - `md_since_norm`、`tail_rank_norm`、near / longtail flags
  - `tvt_dense` / `tvt_densew` / `tvt_dense50` の drift、absolute drift、slope、roughness
  - dense family std / range
  - `likpf_mean` / `beam_mean` / `pf_ancc` と `tvt_densew` の差
  - `pf_vs_dense`、`dense_std`、high-disagreement proxy、longtail interaction

## 結果

Kaggle train v1 を完了した。Kernel は `kentookumura/exp151-tvt-dense-addonly-exp092-train` version 1。出力は `experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/kaggle/output/train_v1`。

| model | pooled RMSE | exp092 reference | delta |
| --- | ---: | ---: | ---: |
| lgb0 | 9.597243443 | 9.533126438 | +0.064117005 |
| lgb1 | 9.355161771 | 9.322479896 | +0.032681876 |
| lgb2 | 9.375504593 | 9.338192405 | +0.037312188 |
| lgb_mean | 9.388714996 | 9.343064066 | +0.045650930 |

Best は `lgb1` だが exp092 best を上回らない。`lgb_mean` も悪化したため anchor 更新にはならない。

Distance bucket は `lgb1` で near `000_050` RMSE 1.319889、`1000_plus` RMSE 10.264017。worst wells は `86454a6f` RMSE 57.602753、`fb03ae90` 40.862579、`1b1eba53` 39.959881 で、global OOF 改善を伴わない。

Feature importance では `tdc_dense_std_norm`、`tdc_tail_rank_norm`、`tdc_abs_tvt_dense50_minus_tvt_densew_norm`、`tdc_high_disagreement_proxy`、`tdc_likpf_mean_minus_tvt_densew_norm` が上位に入った。dense confidence signal は使われているが、exp092 surface に add-only するだけでは過学習またはノイズ増加が勝つ。

## 判断基準

exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を baseline とする。global OOF が改善しても、near-row、longtail、worst-well regression、exp115 hidden-like stress、raw-test/full-train feature parity が崩れる場合は submit しない。

## 判断

`tvt_dense_addonly_confidence_features_on_exp092` は train-side rejected。inference port / submit は行わない。今後 dense 候補を使う場合は、exp135 と exp151 の結果から、全 row の add-only feature ではなく `segment_level_dense_candidate_verifier` のような low-switch / segment-level 診断に限定する。
