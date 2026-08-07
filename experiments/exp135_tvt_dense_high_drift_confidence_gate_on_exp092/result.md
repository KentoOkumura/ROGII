# exp135_tvt_dense_high_drift_confidence_gate_on_exp092 結果

## 状態

Kaggle train v2 完了。診断専用のため inference / submit は行わない。

## 仮説

`tvt_dense` 系候補は PF / ML 共通 worst regime の oracle headroom を持つ可能性がある。ただし direct replacement は危険なので、exp092 `lgb1` を base に、high-drift / high-disagreement well または segment だけで低頻度に使う gate を train-side OOF で評価する。

## 評価方針

LightGBM の新規学習は行わない。exp092 / exp073 の saved OOF prediction と exp072 feature cache を固定入力にして、target-free gate 条件だけで posthoc 予測を作る。

比較基準:

- exp092 `lgb1`: RMSE 9.322479896 / within10 0.822047 / Public LB 8.350
- exp073 `lgb_mean`: RMSE 9.526374749
- `likpf_mean`: RMSE 11.594897672
- `tvt_dense` 系 single candidate / oracle headroom

## 結果

Kaggle train v2 は `kentookumura/exp135-tvt-dense-gate-train` で完了した。output は `experiments/exp135_tvt_dense_high_drift_confidence_gate_on_exp092/kaggle/output/train_v2`。

全体では exp092 base が最良で、すべての non-oracle gate は悪化した。

| variant | RMSE | delta vs exp092 | within10 | max well regression |
| --- | ---: | ---: | ---: | ---: |
| `base_exp092_lgb1` | 9.322480 | 0.000000 | 0.822047 | 0.000000 |
| `seg_dense50_q75_tail1000_min100_clip20_a050` | 9.874846 | +0.552366 | 0.788303 | +9.535752 |
| `seg_densew_q75_tail1000_min100_clip20_a050` | 9.891295 | +0.568815 | 0.786903 | +9.547818 |
| `well_dense_q90_tail1000_rate10_clip20_a050` | 10.003634 | +0.681155 | 0.775691 | +9.850307 |
| `single_tvt_dense50` | 19.994558 | +10.672078 | 0.675590 | +150.497319 |
| `single_tvt_densew` | 20.102567 | +10.780087 | 0.672133 | +150.858963 |
| `single_tvt_dense` | 23.470396 | +14.147916 | 0.582951 | +197.092310 |

Oracle では headroom がある。all candidate oracle は RMSE 3.421444、dense candidate oracle は RMSE 10.971366。ただし dense oracle ですら exp092 base より +1.648886 悪く、直接 gate 条件としては使えない。

PF `likpf_mean` worst50 では dense 候補が強く、`single_tvt_densew` は RMSE 17.505173 で exp092 22.157400 から -4.652228 改善した。gate も `seg_dense_q75_tail1000_min100_clip20_a050` で -3.129011 改善した。

一方、exp092 worst50 では single dense は大きく壊れる。clipped gate は exp092 worst50 を -0.90 程度改善するが、全体 RMSE と near / mid buckets を悪化させる。`seg_dense50_q75_tail1000_min100_clip20_a050` は near `000_050` を +0.082539、`050_100` を +0.172969、`1000_plus` を +0.554533 悪化させた。

最大 by-well regression も大きい。best-looking gate `seg_dense50_q75_tail1000_min100_clip20_a050` では `389ae58f` +9.535752、`059c8f24` +9.300540、`99529c45` +9.278271、`071d7b45` +9.055897、`b0d42b0d` +8.985595 と悪化した。

raw-test parity checklist は feature column availability と target-free gate 条件について pass。ただし inference port は未実装で、この結果から port しない。

## 解釈

`tvt_dense` は PF worst / common worst を救う候補としては有効だが、現行の target-free high-drift / high-disagreement gate では選択精度が足りない。global OOF、within10、near-row、worst-well regression を同時に壊すため、`tvt_dense_high_drift_confidence_gate_on_exp092` は rejected とする。

## 次

- exp135 の gate は inference port / submit しない。
- `tvt_dense` を使うなら、hard gate ではなく `segment_level_dense_candidate_verifier` のような低頻度 segment verifier / selector へ下げる。
- exp092 への単純 dense correction ではなく、ML add-only confidence feature または candidate verifier feature として扱う。
