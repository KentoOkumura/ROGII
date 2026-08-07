# exp233_adaptive_outlier_mixture_likelihood_pf 結果

## 結論

`mix_eps_0p02` と `mix_eps_0p05` はいずれも full eligible-well surface（773 wells /
3,783,989 rows）を完走したが、exp209 から復元した exp072 `likpf_mean` control を大きく
下回った。overall、`1000_plus`、worst-well guard がともに不合格のため、**train-side
不採用**とする。raw-test regeneration、inference、submission は行わない。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 比較: exp209 reconstructed exp072 `likpf_mean` Gaussian control、exp232 temperature-only variants
- 変更: gate-on row のみ state-neutral Uniform-GR outlier mixture
- variants: `epsilon=0.02`、`epsilon=0.05`
- 固定: gate、500 particles、128 seeds、transition、resampling、seed mean aggregation
- Uniform support: GR `[0,500]`

## 判定条件

overall RMSE だけでなく、1000+、hidden-like、worst-well、ESS、resampling、gate/mixture
rate、sampled particle interval coverage、first-loss を確認する。exp232 の id-aligned
artifacts がない並行初回 run は comparison pending と記録し、採用しない。

## 結果

| variant | Kaggle version | runtime | RMSE | control差 | 1000_plus RMSE | 最大 well 悪化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp072 `likpf_mean` control | - | - | 11.594898 | 0.000000 | 12.704015 | 0.000000 |
| `mix_eps_0p02` | canonical v3 | 32,370s | 13.519963 | +1.925065 | 14.759210 | +44.485364 |
| `mix_eps_0p05` | canonical v4 | 24,346s | 13.550173 | +1.955275 | 14.799418 | +46.711288 |

- ε=0.02 が mixture 内では僅かに小さい悪化だが、baseline に対して RMSE を +1.925065 悪化させ、long-tail と worst-well も不合格である。
- ε=0.05 は exp232 `temp_t2` の checkpoint-free v2 と exp072 input / exp209 control / schema content SHA が一致する。ID-aligned 比較でも ε=0.05 は T=2 より RMSE が +0.020286 悪い。
- gate は target-free のまま（all-seed 659 / 677 rows、any-seed 4,641 / 4,662 rows）で、控えめな mixture 率にしても長期 path の回帰を抑えられなかった。

## 判断と次のアクション

state-neutral Uniform-GR mixture による direct PF observation update 枝を閉じる。epsilon の追加 grid、temperature との同時変更、global mixture、raw-test regeneration、inference、submit は行わない。robust likelihood を再検討する前に、`adaptive_likelihood_pf_trajectory_containment_audit` で gate event 後の cumulative path divergence、ESS、resampling、seed disagreement を診断し、長期回帰の発生点を特定する。
