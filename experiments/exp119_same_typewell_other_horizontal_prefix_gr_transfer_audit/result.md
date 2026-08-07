# exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit 結果

## 仮説

同じ native typewell overlap group に属する他 horizontal well の source prefix raw GR と `TVT_input` は、query evaluation-zone の raw GR に対応する弱い TVT path prior になり得る。

## 設定

- 親: `exp109_typewell_neighbor_prior_features`
- 参照: `exp065_typewell_supertype_cluster_cv_audit`、`exp099_pf_multi_observation_likelihood_probe`
- 検証: well-grouped 5 folds train pseudo-tail OOF prefix GR transfer audit
- メトリック: RMSE / MAE / within10 / match score / signal correlation / bucket / by-well
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV | best `likpf_mean` RMSE 11.594897672 |
| Public LB | - |
| Private LB | - |

Kaggle train v1 は `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` version 1 として実行したが、約 2088 秒で `DeadKernelError: Kernel died`。生成物は support files と log のみで、metrics / OOF artifacts は得られなかった。

v2 では source / candidate grid を縮小し、同じ kernel id の version 2 として実行完了。3,783,989 rows / 773 wells を評価した。output download は大きい OOF gzip で接続が切れたため `summary.json` / `signal_metrics.csv` / OOF gzip は未取得だが、candidate / bucket / by-well metrics は取得済み。

| candidate | RMSE | MAE | within10 | delta vs likPF RMSE |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean` | 11.594897672 | 7.067632584 | 0.772807479 | 0.000000000 |
| `different_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` | 11.607097336 | 7.219326819 | 0.771474230 | +0.012199664 |
| `same_typewell_random_control_slope_likpf_mean_corr_a0p1_c20` | 11.609221461 | 7.222340473 | 0.771488501 | +0.014323789 |
| `same_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` | 11.614959308 | 7.222329239 | 0.771515456 | +0.020061636 |

same-typewell GR match の最良補正は baseline より悪く、さらに different-typewell control / same-typewell random control よりも悪い。期待した「同じ typewell の他 horizontal prefix GR を query evaluation zone に転用する」信号は確認できなかった。

by-well では `same_typewell_gr_match_slope_likpf_mean_corr_a0p1_c20` が 291 wells 改善 / 407 wells 悪化 / 75 wells 同値。最大悪化 +1.907160 RMSE、最大改善 -1.903311 RMSE、平均 delta +0.120090 RMSE。global だけでなく well 単位でも採用根拠は弱い。

## 再現性

- deterministic anchor: false
- seed policy: deterministic well fold assignment with fixed seed 42。random control は SHA256 由来 index。
- kernel version: old id `kentookumura/exp118-same-typewell-prefix-gr-transfer-train` v1 failed / v2 completed
- feature cache SHA: summary 未取得のため未記録
- cluster assignment SHA: summary 未取得のため未記録
- candidate metrics SHA: `7ddc6a69369f122440a7a01c386576f002abcd077e2a4afd316c504b57515914`
- bucket metrics SHA: `6acaa0ea97ed763fb408a74f645b1ab1a142543be41ffda6a0a882a8b7bb7198`
- by-well metrics SHA: `dd583e1db065dd4dce365f645e948a5b8b478273a4010975057e1d5192341b45`
- feature schema SHA: `148a1df790b142e2e9731d000501349a29887e35453936d7b1cf721263ea67f3`
- OOF prediction SHA: 未取得
- model SHA / manifest SHA: model なし
- submission SHA: submission なし
- rerun result: 未実行

## 解釈

v1 は不採用。GR matching の full grid が CPU notebook の実行量またはメモリを超えた可能性が高い。

v2 の結果も negative。same-typewell prefix GR transfer は `likpf_mean` を改善せず、negative controls より弱い。raw GR waveform を cross-horizontal に直接対応付ける方向は、exp008/017/042 の悪化履歴と整合して弱い。

## 次

`same_typewell_other_horizontal_prefix_gr_transfer_audit` は完了。direct correction / candidate path / inference port はしない。今後この情報を使う場合は、GR match score、coverage、source count を quality diagnostic として別の confidence / add-only feature 実験に限定する。
