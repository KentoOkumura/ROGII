# exp201_typewell_spatial_tvt_error_readout 結果

## 仮説

exp148 OOF の残差について、共通 typewell group、XY 近傍、急激な true TVT step、well 全体の予測 offset に傾向があるかを診断する。

## 設定

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 入力: exp148 train v1 `lgb_mean` OOF prediction
- filter: `variant=learned_likelihood_confidence_addonly`, `mode=gpu_repro_guard_dp_threads8`, `model=lgb_mean`
- typewell: `native_overlap_1` の exact typewell group。1 well は 1 typewell group のみに属する。
- 実行: Kaggle CPU `kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a` v2
- 目的: diagnostic only。新規 model training、inference、submission はなし。

## 全体

| 項目 | 値 |
| --- | ---: |
| rows | 3,783,989 |
| wells | 773 |
| typewell groups | 54 |
| exp148 OOF RMSE | 8.501281182 |
| MAE | 5.335651346 |
| mean bias `pred - true` | -0.007505 |
| offset wells | 66 / 773 |
| offset well rate | 8.54% |

## 主な発見

### 共通 typewell

同じ typewell を使っている well 同士で、残差プロファイル形状が一貫して似るという強い傾向は見えなかった。100-bin residual profile の pairwise correlation は、全 pair median -0.0019、same-typewell median 0.0037、XY 近傍 median 0.0082 で、ほぼゼロ付近だった。

一方で、typewell group 単位の「高 RMSE / offset 多発」hotspot は明確にある。特に `native_overlap_1_cluster_0034` は RMSE 23.125、abs error mean 15.786、7 wells 中 3 offset。`native_overlap_1_cluster_0033` は RMSE 19.775、abs error mean 13.069、8 wells 中 4 offset。typewell group RMSE は offset well rate と相関 0.706、well_abs_bias_mean と相関 0.910 だった。

結論として、typewell は residual shape correction というより、offset / high-risk group の検出 signal として使う方が筋が良い。

### XY 近傍

XY が近い well 同士で bias sign や residual shape が揃う傾向は弱い。8-nearest の bias sign 一致率は 0.4945、全 pair は 0.5009 で差がない。nearest bias abs diff mean 6.239 も all-pair mean 6.260 とほぼ同じ。

高 error well はむしろ局所近傍から浮いている例が多い。top RMSE の `86454a6f` は self bias -43.49 ft だが、8-neighbor bias mean は -0.19 ft。`1b1eba53` は self bias -42.74 ft に対して neighbor bias mean +6.19 ft。`fb03ae90` は self bias +41.67 ft に対して neighbor bias mean -0.65 ft。

結論として、XY 近傍平均で直接補正するのは危険。使うなら「周辺と自分が乖離している outlier signal」として使うのがよい。

### 全体 offset well

offset well は 66 wells、内訳は underpredict 37、overpredict 29。top30 high-error wells のうち 27 wells が offset flag で、well RMSE と abs_bias の相関は 0.948。exp148 の worst error は、局所的な揺れよりも whole-well offset が主因。

offset は単一方向に偏っておらず、typewell group 内でも over / under が混在する group がある。たとえば `cluster_0003` は offset 5 wells だが over 3 / under 2、`cluster_0008` は over 2 / under 3。`cluster_0033` と `cluster_0034` は高率 offset group だが、方向固定の rule にはしにくい。

結論として、次に試すなら「offset 検出 confidence / sample weight / router」であり、typewell group だけから固定方向に加算・減算する postprocess は避ける。

追加で offset 方向の揃い方を確認した。offset wells が 2 本以上あり、その offset 方向が全て一致する group は 4 つだけだった。

| typewell group | wells | offset wells | offset rate | offset direction | offset bias mean | all-well same sign rate |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `native_overlap_1_cluster_0004` | 38 | 4 | 0.105 | underpredict | -15.288 | 0.500 |
| `native_overlap_1_cluster_0029` | 10 | 2 | 0.200 | underpredict | -18.968 | 0.600 |
| `native_overlap_1_cluster_0009` | 28 | 2 | 0.071 | underpredict | -17.565 | 0.536 |
| `native_overlap_1_cluster_0013` | 22 | 2 | 0.091 | underpredict | -15.605 | 0.591 |

このうち、group 全体の bias 方向も少し揃っている候補は `cluster_0029` だけ。ただし 10 wells 中 6 over / 4 under で、offset wells 2 本はいずれも underpredict という状態なので、強い rule というより弱い test-only prior と見るべき。

一方で group 全体の bias 方向が 75% 以上揃っている group は `cluster_0032`、`cluster_0036`、`cluster_0022`、`cluster_0006`、`cluster_0030` だが、これらは offset wells が 0-1 本で、強い offset 補正対象ではない。

### 急激な TVT 上昇/下降

sharp true TVT step は上位 0.5% 閾値 `abs(true_step) >= 0.08984375` で、20,764 rows、481 wells。上昇 10,361 rows、下降 10,403 rows で、方向の偏りはほぼない。

sharp step では `abs(pred_step) / abs(true_step)` の平均が 3.337 と大きく、true の小さな段差に対して予測 step が大きく振れやすい。方向別では down 3.571、up 3.102。平均 abs step error は 0.413 ft。件数が多い group は `cluster_0009`、`cluster_0011`、`cluster_0007`。ratio が高い group は `cluster_0014`、`cluster_0035`、`cluster_0034`。

結論として、急変部は全体 RMSE の主因というより、特定 group/well の局所不安定性 signal として使うのがよい。offset well 対策とは分けて扱うべき。

## 生成物

- `metrics.json`
- `artifacts/well_error_profile_summary.csv`
- `artifacts/typewell_group_metrics.csv`
- `artifacts/xy_neighbor_bias_similarity.csv`
- `artifacts/well_residual_profiles_100bin.csv`
- `artifacts/offset_wells.csv`
- `artifacts/high_error_wells_top30.csv`
- `artifacts/tvt_sharp_step_rows.csv`
- `artifacts/tvt_sharp_step_wells.csv`
- `artifacts/tvt_sharp_step_typewell_metrics.csv`
- `artifacts/typewell_offset_direction_summary.csv`
- `artifacts/typewell_group_bias_rmse_top.svg`
- `artifacts/xy_bias_map.svg`
- `artifacts/offset_residual_profiles.svg`
- `artifacts/sharp_step_true_vs_pred.svg`

同じ生成物を Kaggle output として `kaggle/output/train_v2/` に取得済み。

## 次

直接補正ではなく、exp148 系 ML route の add-only feature として次を検討する。

1. well が自分の XY 近傍 bias 分布からどれだけ外れているかを表す outlier confidence。
2. typewell group の historical offset rate / high-RMSE flag / sharp-step instability flag。
3. whole-well offset を検出するが、方向固定補正はしない sample weight または uncertainty feature。

この readout だけでは submission 候補は作らない。
