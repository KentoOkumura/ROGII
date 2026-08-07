# exp135_tvt_dense_high_drift_confidence_gate_on_exp092

## 状態

- ルート: ml_model
- 状態: completed_train_side_rejected_no_submit
- CV: exp092 base 9.322479896、best gate 9.874846008
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-26
- 親実験: exp092_u_projection_correction_disagreement_fullrun

## 仮説

PF / ML 共通 worst well や `likpf_mean` worst well では、`tvt_dense` 系候補が oracle best になる比率が高い。ただし `likpf_mean` や PF/Beam direct replacement、broad hard switch は悪化済みなので、`tvt_dense` は全体置換ではなく high-drift / high-disagreement regime の低頻度 gate としてだけ評価する。

## 変更点

- exp092 `lgb1` OOF prediction を base として固定する。
- exp073 `lgb_mean` OOF prediction を reference として読む。
- exp072 full replay feature cache から `tvt_dense` / `tvt_densew` / `tvt_dense50` / `tvtF_ANCC` と PF/Beam/likPF 候補を読む。
- LightGBM の新規学習は行わない。
- `dense_std`、`tvt_dense_d`、PF-dense disagreement、exp092-dense disagreement、PF/Beam disagreement、tail rank を使った target-free segment / well gate を小 grid で比較する。
- inference / submission は作らない。

## 検証方針

- Fold: upstream exp092 / exp073 の OOF prediction を固定入力として使う。
- Group: well
- 主指標: RMSE
- 補助指標: within10、gate rate、common PF+ML worst 26 wells、PF `likpf_mean` worst50、tail bucket、near-row bucket、path continuity、max well regression。
- Leakage Check: `target_tvt` と oracle は評価専用。gate 条件には使用しない。

## 実行入口

- 学習 notebook: `exp135_tvt_dense_high_drift_confidence_gate_on_exp092_train.ipynb`
- 推論 notebook: `exp135_tvt_dense_high_drift_confidence_gate_on_exp092_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp135_tvt_dense_high_drift_confidence_gate_on_exp092 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp135-tvt-dense-gate-train --title 'exp135 tvt dense gate train' --run-on-push --strict"`

## 生成物

- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_metrics.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_gate_variants.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_by_well.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_bucket_metrics.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_common_worst_metrics.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_rawtest_parity_checklist.csv`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_prediction_sample.csv.gz`
- `artifacts/exp135_tvt_dense_high_drift_confidence_gate_on_exp092_summary.json`

## 所見

Kaggle train v2 完了。LightGBM を再学習せず、exp092 OOF と exp072 dense candidate cache だけで posthoc gate を評価した。

全体では exp092 base が最良で、best gate `seg_dense50_q75_tail1000_min100_clip20_a050` でも RMSE 9.874846、exp092 から +0.552366 悪化した。PF `likpf_mean` worst50 では dense 系が改善するが、near-row、overall、worst-well regression を壊すため inference port / submit はしない。

## 注意

- exp135 は deterministic submission anchor ではない。
- OOF 改善が出ても、near-row、worst-well regression、path continuity、raw-test parity が通るまで inference port しない。
- 今回は OOF 改善が出ず、best gate でも max well regression +9.535752 のため rejected。
