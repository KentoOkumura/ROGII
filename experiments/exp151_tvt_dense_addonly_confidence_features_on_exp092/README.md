# exp151_tvt_dense_addonly_confidence_features_on_exp092

## 概要

`exp092_u_projection_correction_disagreement_fullrun` の U-projection correction / disagreement LightGBM surface に、`tvt_dense` family の target-free confidence 特徴を add-only で足す実験。

`tvt_dense` / `tvt_densew` / `tvt_dense50` を prediction replacement や hard switch には使わない。dense candidate の drift、slope、roughness、dense family disagreement、PF/Beam/likPF と dense の差、near / longtail interaction を LightGBM の補助特徴として渡す。

## 仮説

exp135 では dense hard gate は global OOF を大きく壊したが、PF `likpf_mean` worst50 と common PF+ML worst26 では `tvt_densew` 単体が exp092 より大きく良い場面があった。dense surface は「採用する予測」ではなく、exp092 が外れやすい regime の説明変数として使える可能性がある。

## 検証方針

GroupKFold by well の full-row exp092 surface 上で `tvt_dense_confidence_addonly` を学習する。既存 exp092 metrics を baseline とし、control 再学習は明示承認なしに行わない。

初回 Kaggle train 対象は `tvt_dense_confidence_addonly` 1 variant、LightGBM 3 config、5 folds、合計 15 boosters。

## 比較

- baseline: `exp092_u_projection_correction_disagreement_fullrun` `lgb1` CV 9.322479896 / Public LB 8.350
- negative reference: `exp135_tvt_dense_high_drift_confidence_gate_on_exp092` best dense gate RMSE 9.874846
- references: `exp127_learned_likelihood_features_on_exp092`、`exp130_pfbeam_normalized_diagnostic_score`

## 状態

- ルート: MLモデル
- 状態: completed_train_side_rejected_no_submit
- CV: best `lgb1` 9.355161771
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-27
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`

## 所見

Kaggle train v1 完了。best は `lgb1` RMSE 9.355161771 で、exp092 `lgb1` 9.322479896 から +0.032681876 悪化した。`lgb0`、`lgb2`、`lgb_mean` もすべて exp092 同 model より悪化したため、inference port / submit はしない。

追加した `tdc_*` features は feature importance 上位に入ったが、OOF 改善にはつながらなかった。dense surface は row-wise add-only feature ではなく、segment-level verifier / selector 側で低頻度に扱う候補として残す。
