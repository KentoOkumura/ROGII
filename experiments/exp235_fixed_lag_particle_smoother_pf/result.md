# exp235_fixed_lag_particle_smoother_pf 結果

## 結論

exact 4 deterministic well shard を strict merge した `lag64` は、同一行の exp072
`likpf_mean` control より明確に悪化した。overall RMSE と within-10 の採用条件を満たさない
ため、**lag64 は train-side 不採用**とする。raw-test inference と submission は行わない。

ユーザー判断により `lag128` / `lag256` は実行しない。この fixed-lag PF 枝を閉じ、raw-test
inference と submission には進まない。

## 実行と完全性

- variant: `lag64`、500 particles、128 seeds、forward transition / Gaussian likelihood /
  resampling は exp072 と同一
- 実行: 4 deterministic well shards、773 wells、3,783,989 rows
- shard runtime 合計: 53,562.860 秒（約 14.88 CPU hours）
- strict merge: shard manifest の full-well union、well 重複なし、全 row ID 重複なしを確認
- merged row candidates decompressed SHA256:
  `6376acff1762b438c0bf173da3fc8c3fc6feebad692d1e3b4eb2628b0c0ae0e5`
- merged artifact:
  `artifacts/lag64_merged_v3/exp235_fixed_lag_particle_smoother_pf_merged_row_candidates.csv.gz`

logical shard 0 / 2 は stale dedicated notebook の代わりに既存 shard1 / shard3 kernel の v2 を
器として使った。採否の根拠は notebook title ではなく、各 summary の
`execution.well_shard_index`、target-well manifests、row IDs と merge manifest である。

## 全体結果

| candidate | RMSE | control差 | MAE | within-10 |
| --- | ---: | ---: | ---: | ---: |
| exp072 `likpf_mean` | 11.594898 | 0.000000 | 7.067633 | 0.772807 |
| `pf_lag64_mean` | 13.495448 | +1.900550 | 9.067267 | 0.673755 |

- fixed-lag smoother は RMSE を約 16.4% 悪化させ、within-10 を 9.91 percentage points 下げた。
- 順方向 PF、particle 数、seed 数、resampling、観測尤度を固定したため、この差は future raw GR
  を用いる fixed-lag ancestor smoothing rule の差として読める。
- `lag64` の後により重い lag を走らせても、現時点では採用経路や inference へ進む根拠はない。

## 原因分析の制約

- 全 distance bucket で悪化し、特に `1000_plus` は RMSE 12.702990 から 14.738212
  （+2.035222）になった。末尾 fallback の一部だけが原因ではない。
- ただし exp235 の particle seed は `stable_seed(EXPERIMENT_NAME, well, variant, ...)`、frozen
  exp072 control は `stable_seed("likpf", "train", well)` であり、同じ forward PF 乱数列ではない。
  実際、各 well の最後64行（fixed-lag を適用せず exp235 forward estimate を残す行）でも RMSE は
  18.449056、exp072 control は16.682135で、+1.766921差がある。この 49,472行は全体の1.3%だけで
  全体悪化を説明しないが、smoothing だけの因果比較ではない。
- したがって「実装した lag64 candidate は不採用」は確定だが、ancestor smoothing rule 単独の
  効果を厳密に測るには、exp072 と完全に同じ seed の forward candidate を同一 run に保存する
  paired ablation が必要である。ユーザー判断により、この追加実行は行わない。

## 閉鎖判断

`lag128` / `lag256` の実行、seed-paired 再監査、raw-test inference、submission は行わない。
HMM はすでに exact forward-backward smoother で全 future raw GR を使っているため、HMMへ単に
fixed lag を加えることは新しい情報の追加ではなく、既存 exact smoothing の近似・正則化である。
