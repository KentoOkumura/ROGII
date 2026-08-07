# exp232_adaptive_robust_likelihood_pf 結果

## 状態

`temp_t2` と `temp_t4` の独立 Kaggle CPU kernel はともに 773 wells / 3,783,989 rows を完走した。両方とも exp209 から復元した exp072 `likpf_mean` control を大きく下回り、long-tail と worst-well guard も破ったため、**train-side 不採用**とする。raw-test regeneration、inference、submission は実施しない。

## 固定設定

- control: exp209 enriched cache から復元する exp072 `likpf_mean`（`T=1`）
- variant: `temp_t2` / `temp_t4`
- PF: 500 particles x 128 seeds、raw GR/typewell GR、既存 transition / resampling、seed mean aggregation
- gate: high innovation + target-free corroborating signal
- out of scope: outlier mixture、global temperature、particle reinjection、inference、submission

## 判定基準

overall RMSE だけでなく、sampled particle p05-p95 coverage、first sampled loss、`1000_plus`、exp115 hidden-like、worst-well regression、ESS/resampling と gate rate を確認する。coverage 改善と RMSE / worst-well guard の両方を満たすまで、raw-test regeneration・inference・submit へ進まない。

## 結果

| variant | kernel | runtime | RMSE | control差 | 1000_plus RMSE | 最大 well 悪化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exp072 `likpf_mean` control | exp209 reconstructed | - | 11.594898 | 0.000000 | 12.704015 | 0.000000 |
| `temp_t2` | `exp232-adaptive-robust-pf-t2` v2（checkpoint-free output recovery） | 39,327s | 13.529887 | +1.934989 | 14.775089 | +45.905685 |
| `temp_t4` | `exp232-adaptive-robust-pf-t4` v2 | 34,207s | 13.532730 | +1.937833 | 14.778864 | +45.706171 |

- gate は疎だった（全 seed 共通発火: T=2 で 685 rows、T=4 で 715 rows）が、いずれも overall、`1000_plus`、hidden-like、worst well で control より悪化した。
- interval は候補のみを測定したため control との coverage 改善は主張しない。sampled coverage は T=2 で 0.219654、T=4 で 0.219931 に留まった。
- T=2 と T=4 はともに不採用で、より小さい overall 悪化の T=2 も後続候補にはしない。
- T=2 の v2 は exp233 `mix_eps_0p05` v4 と input / control / schema content SHA が一致する。ID-aligned 比較でも mixture の RMSE 13.550173 は T=2 より 0.020286 悪く、mixture を温度の代替として採用する根拠はない。

## 判断と次のアクション

固定温度による gated likelihood 緩和は、発火が稀でも resampling 後の軌跡に広い回帰を生むため、この direct PF observation update 枝を閉じる。後続で robust likelihood を再検討する場合は、まず gate 発火後の累積 path divergence / RMSE delta を target-free event 単位で監査し、control を越える長期的回帰がないことを前提条件にする。exp233 mixture は別仮説として実行記録を保つが、exp232 の温度 variant を混ぜたり、温度単独の再 grid は行わない。

exp209 cache の exp072 v2 full artifact との exact parity は未証明である。さらに split run は exp072 input cache を別の Kaggle dataset copy から解決しているため、両 variant 間の細かな subgroup 差は exact parity evidence として扱わない。ただし同一 control 指標に対する約 +1.93 RMSE の悪化は、採否判断に十分明確である。復元値は gate や PF 更新には使わず、比較列だけに限定した。
