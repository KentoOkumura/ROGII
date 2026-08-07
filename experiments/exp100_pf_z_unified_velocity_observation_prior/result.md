# exp100_pf_z_unified_velocity_observation_prior 結果

## 状態

Kaggle train v2 完了。提出なし。

## 仮説

`pf_z` の粒子速度 prior に XY trajectory と prefix slope を弱く足し、GR observation likelihood に prefix-fitted affine calibration を入れることで、train pseudo-tail の PF 単体 RMSE、within10、path smoothness が改善する可能性を検証した。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 検証: train well pseudo-tail PF z-prior Stage 1 ablation
- Score rows: raw train horizontal well の `TVT_input` missing rows
- rows: 3,783,989
- wells: 773
- Control: `pf_z_control`
- Variant: XY slope、prefix slope、GR calibration とその組み合わせ 7 本
- 粒子数: 360
- seed: experiment / variant / well id から stable SHA256 seed

## 結果

- Kernel: `kentookumura/exp100-pf-z-unified-prior-train` v2
- status: `KernelWorkerStatus.COMPLETE`
- runtime: 3,986.42 sec
- output: `experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/output/train_v2`

| variant | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `pf_z_xy_slope` | 29.404162 | 10.959212 | 0.655593 | -0.981801 |
| `pf_z_xy_plus_prefix` | 30.157085 | 20.090660 | 0.439449 | 17.236638 |
| `pf_z_xy_plus_gr_calibrated` | 31.700123 | 11.835740 | 0.637807 | -1.231632 |
| `pf_z_xy_prefix_gr_calibrated` | 38.796222 | 25.383146 | 0.389975 | 22.906418 |
| `pf_z_prefix_slope` | 107.007015 | 60.600226 | 0.224214 | 9.172049 |
| `pf_z_prefix_plus_gr_calibrated` | 135.734575 | 81.645557 | 0.179921 | 10.420694 |
| `pf_z_control` | 163.301551 | 92.155466 | 0.179258 | -23.557848 |
| `pf_z_gr_calibrated` | 196.811290 | 121.072003 | 0.137380 | -30.588475 |

best は `pf_z_xy_slope`。standalone control からは RMSE -133.897390、within10 +0.476334 改善した。

ただし、best RMSE 29.404162 は既存 PF/Beam 候補の `likpf_mean` RMSE 11.594897 より大きく弱い。`pf_z_xy_slope` を直接 inference port / submit 候補にはしない。

## 注意

この実験の `pf_z_control` は exp100 の standalone rerun 実装内 control であり、exp072 の保存済み `pf_z` との厳密 parity control ではない。実装上、velocity fit は unified design を使っているため、結果は「exp100 の unified PF rerun 内比較」として扱う。真の exp072 parity を確認するには、z-only coefficient fit を既存 public replay と完全一致させた別 smoke が必要。

## 再現性

- deterministic anchor: false
- seed policy: stable SHA256 seed from experiment / variant / well id
- PF stochastic components: particle initialization、process noise、resampling
- model SHA / submission SHA: model と submission は作らない
- `candidate_wide.csv.gz` raw SHA: `2a6f24058b7b026248661cfd408aa8ada12d8183882b1299708f4393cc39595c`
- `candidate_wide.csv.gz` decompressed SHA: `0de40234e68e1cf956da0fafa08323a828865157052eb3af616faa677b0a0389`
- `candidate_long.csv.gz` raw SHA: `78c8524ea113b4ddfc7a3998c0031d8d4dbd1adf11bcbe6d7cfbdae364fcca22`
- `candidate_long.csv.gz` decompressed SHA: `591675720c4f3af1c3368faab95c2ecf0bb31dbee62ee0c220271ffb61fe4139`
- `variant_metrics.csv` SHA: `bc64dc86fa500721000b395fe8b70da51bde79d38b766063e0cdeac7ef7a0ccd`
- `summary.json` actual local SHA: `a862b5d8d327e8388507baec9b4b927dddc3e6e9fe36e2e68c92e8960b4fa721`

v2 の `summary.json` 内に記録された summary 自身の SHA は自己参照のため final file SHA と一致しない。生成物 CSV / gzip の SHA は summary 記録とローカル検査が一致した。post-run で helper を修正し、次回以降は summary 自身の SHA を記録しない。

## 解釈

XY velocity prior は standalone control からは強く改善したが、絶対性能が既存候補より弱い。prefix slope と GR calibration は単独でも組み合わせでも改善せず、むしろ悪化した。Stage 2 の dense prior / multi-scale GR / acceleration penalty に進む根拠は不足している。

## 次

`pf_z_unified_velocity_observation_prior` は backlog から外す。直接 inference port はしない。PF 側を続けるなら、まず exp072 public replay と strict parity の `pf_z_control` を再現できるかだけを小さく確認し、改善検証はその後に限定する。
