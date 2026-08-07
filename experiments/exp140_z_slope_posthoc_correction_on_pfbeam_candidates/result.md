# exp140_z_slope_posthoc_correction_on_pfbeam_candidates 結果

## 状態

Kaggle train v2 完了。train-side audit として不採用。

## 仮説

Z-driven に見える一部 well では、固定 PF/Beam 候補の slope と `-dZ/dMD` の gap を小さく累積補正することで、`likpf_mean` などの候補を局所的に改善できる可能性がある。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- 検証: train well pseudo-tail posthoc candidate audit
- 主比較: `likpf_mean`
- base candidates: `likpf_mean`, `pf_ancc`, `beam_mean`
- auxiliary: `pf_z`
- 代表 well: `91b301ce`, `ba48188d`, `fef8af96`, `1b1eba53`

## 結果

- 実行: `kentookumura/exp140-z-slope-pfbeam-train` v2
- 行数 / well 数: 3,783,989 rows / 773 wells
- variant 数: 432
- baseline `likpf_mean`: RMSE 11.594897672、MAE 7.067632584、within10 0.772807479
- best overall: `likpf_mean`
- best Z-slope variant: `zsl_likpf_mean_a0p1_c10_z0p1_d5_pfz_agree`
- best Z-slope RMSE: 11.597150618
- delta vs `likpf_mean`: +0.002252946 RMSE
- best Z-slope within10: 0.772796115
- max well regression vs `likpf_mean`: +0.204513805 RMSE

Z-slope 補正は global で `likpf_mean` を超えなかった。best variant でも RMSE / within10 がわずかに悪化し、well 単位では 20 wells 改善、52 wells 悪化、701 wells 同値だった。

bucket では 500-1000ft 付近に小さい改善 variant があるが、主要な `1000_plus` longtail や `z_abs_top_quartile` では baseline を超えない。代表 well でも `pf_z` が局所的に強いケースはあるものの、target-free な Z-slope posthoc 補正として global に採用できる形ではない。

## 再現性

- deterministic anchor: false
- seed policy: 新規乱数なし
- model SHA / manifest SHA: model なし
- submission SHA: submission なし
- source cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- OOF gzip raw SHA: `94d39ad0ec2755899e319e651a046789091ffc46113dc7f55a6c54a27cf57d1c`
- OOF gzip decompressed SHA: `e638132462b1bfea6fe5aa9a3a6a48278dd1de030d83f6a3eb6fd46d9dc7cc32`

## 次

inference port / submit は行わない。Z / `dZ/dMD` 系は hard posthoc correction ではなく、候補 confidence feature、segment-level verifier、または PF/Beam 本体の transition prior を評価する場合に限定する。
