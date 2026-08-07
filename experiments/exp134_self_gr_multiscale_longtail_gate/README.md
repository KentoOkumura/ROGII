# exp134_self_gr_multiscale_longtail_gate

## 状態

completed_train_side_rejected_no_submit。Kaggle train v1 完了。LightGBM 学習なし、submission なし。

## 仮説

exp090 の `self_gr_core_multiscale` は exp073 control から -0.009557 RMSE の微小改善に留まり、単独推論化には弱かった。一方で half-window 25 の `self_gr_sc25_delta_tvt`、`self_gr_sc25_score`、`self_gr_sc25_l2` は longtail / high PF-dense disagreement regime の補助 confidence になる可能性がある。

## 検証方針

exp072 full replay train feature cache と raw train horizontal GR から self-GR multiscale signal を再生成し、`likpf_mean` baseline と `tvt_dense` low-frequency gate を posthoc に比較する。LightGBM は学習しない。評価は overall RMSE、distance / tail bucket、PF-dense disagreement、self-GR quality bucket、common worst wells、near rows、by-well regression を見る。

## 所見

`self_gr_q75` 条件は self-GR なし dense gate の破壊を抑えたが、best self-GR gate でも RMSE 15.304252 で `likpf_mean` 11.594898 より +3.709355 悪い。common-worst 26 wells では `tvt_dense` の headroom は見えるが、self-GR 条件を足すと改善が弱まり、最大 well regression は +96.835970 RMSE と大きい。

結論: self-GR multiscale longtail gate は直接 gate / inference port / submit / 単独 follow-up には進めない。

## 生成物

想定生成物:

- `exp134_self_gr_multiscale_longtail_gate_metrics.csv`
- `exp134_self_gr_multiscale_longtail_gate_by_well.csv`
- `exp134_self_gr_multiscale_longtail_gate_bucket_metrics.csv`
- `exp134_self_gr_multiscale_longtail_gate_signal_metrics.csv`
- `exp134_self_gr_multiscale_longtail_gate_common_worst_metrics.csv`
- `exp134_self_gr_multiscale_longtail_gate_gate_predictions.csv.gz`
- `exp134_self_gr_multiscale_longtail_gate_feature_schema.csv`
- `exp134_self_gr_multiscale_longtail_gate_summary.json`

## 注意

self-GR 由来情報は confidence-only。直接 TVT 補正、hard replacement、submit 目的の新規実験にはしない。
