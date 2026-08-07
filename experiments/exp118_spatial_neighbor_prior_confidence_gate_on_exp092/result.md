# exp118_spatial_neighbor_prior_confidence_gate_on_exp092 結果

## 仮説

exp114 の spatial neighbor prior は信号としては強いが、direct correction では worst-well regression が大きい。exp092 OOF prediction に対して、prior std、neighbor distance、neighbor count、azimuth mismatch、補正幅で confidence gate をかければ、global 改善を残しつつ悪化 well を抑えられる可能性がある。

## 設定

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- spatial prior 親: `exp114_spatial_neighbor_prior_signal_audit`
- 検証: exp092 OOF prediction への posthoc confidence gate audit
- 対象モデル: `lgb1`
- 対象 prior: `xy_plus_trajectory_shape_k8`, `xy_only_k8`
- 提出: なし

## 結果

Kaggle train v1 は complete。best は `lgb1__xy_only_k8__std_q50_distance_q50__a0p05__c5`。

| メトリック | exp092 baseline | best gate | delta |
| --- | ---: | ---: | ---: |
| RMSE | 9.322479896 | 9.321625436 | -0.000854460 |
| MAE | 5.980980325 | 5.979567419 | -0.001412906 |
| within10 | 0.822047051 | 0.822200857 | +0.000153806 |

best policy は `xy_only_k8` prior を使い、`std_q50_distance_q50` gate で 25.738% の row にだけ補正した。補正は `alpha=0.05`、clip 5 ft なので、実際の correction max は 0.25 ft。

## Bucket

| bucket | baseline RMSE | best RMSE | delta |
| --- | ---: | ---: | ---: |
| 000_050 | 1.378416 | 1.354483 | -0.023932 |
| 050_100 | 1.582904 | 1.563830 | -0.019074 |
| 100_250 | 2.289894 | 2.282672 | -0.007222 |
| 250_500 | 3.542546 | 3.544012 | +0.001466 |
| 500_1000 | 5.224985 | 5.229382 | +0.004397 |
| 1000_plus | 10.229080 | 10.227922 | -0.001158 |

near bucket は改善したが、250-1000 ft はわずかに悪化した。global 改善は小さい。

## Worst-Well / Continuity

- by-well: 192 改善 / 195 悪化 / 386 同値
- max regression: +0.208085 RMSE
- max improvement: -0.208868 RMSE
- mean well delta: -0.001259 RMSE
- path step >=10: baseline 1 / best 1
- path step >=25: baseline 0 / best 0
- correction step >=5: 0
- correction step max: 0.5 ft

max regression は audit threshold 0.25 未満で、path continuity は exp092 baseline と同等。exp114 direct correction の +6.508 RMSE regression は gate により大きく抑えられた。

## 再現性

- Kaggle kernel: `kentookumura/exp118-spatial-gate-exp092-train` v1
- runtime: 2536.951905 秒
- exp114 OOF decompressed SHA: `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`
- exp092 predictions decompressed SHA: `6dc3d53d6cb5621b86360929d638f2a5d853c58ea3e7a53d3da86e614e5f2f69`
- gate metrics SHA: `93d032ad8cfe98589e9a0c094cafd0040f4a6aa3661d154918b1ba914ae25276`
- by-well delta SHA: `38ef3859aef37e00b3728c1aa28db4e2713f61996c12dc460835152e02b753e5`
- bucket metrics SHA: `9dedc19537d2c0dcc613c7b6798b5b35a9c91da2de710467c0e1e13360da53e4`
- path continuity SHA: `872e1b45ba5f1dbf8ffc3aa14f4463a967bebd3273cf8c090021654605c44fe9`
- top gated predictions decompressed SHA: `d3f3a1102d2cf809ef2880acf0298082af80109e4c5b77e1998a6081a6b1b7fe`

`top_gated_predictions.csv.gz` は Kaggle summary には SHA が記録されているが、ローカル output download は 600 秒で timeout し、該当 gzip は 0 byte のまま。採否判断に必要な gate / by-well / bucket / path / summary は取得済み。

## 解釈

confidence gate は有効。exp114 の spatial prior は、exp092 のような強い OOF prediction に対しても、ごく小さい clipped correction なら悪化 well を大きく抑えつつ微小改善を出せる。

ただし改善幅は RMSE -0.000854 と小さく、250-1000 ft bucket はわずかに悪化する。これだけで inference port / submit には進めない。best policy は `confidence_gate_supported_for_review_no_submit` として保持し、提出候補にする場合は raw-test/full-train parity と exp115 hidden-like stress readout を先に確認する。

## 次

direct submit はしない。続けるなら `exp118` 内で同じ best policy の raw-test/full-train parity と hidden-like stress readout を追加するか、別実験として exp092 系 ML add-only feature 化に進める。
