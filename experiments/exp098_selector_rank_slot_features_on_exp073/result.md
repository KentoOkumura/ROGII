# exp098_selector_rank_slot_features_on_exp073 結果

## 仮説

PF/Beam/likelihood-PF 候補集合の情報を rank slot structured features として exp073 LightGBM に渡すと、候補を直接選択するより安全に headroom を使える可能性がある。

## 設定

- 親: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache: `exp072_exp063_full_replay_feature_cache`
- 検証: 5-fold GroupKFold by `well`
- メトリック: RMSE
- シード: 42
- target: `TVT - last_known_tvt`
- 候補: `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`
- 追加特徴量: 64
- 合計特徴量: 260
- active variant: `rank_slot_u_disagreement`

## 実行

- Kernel: `kentookumura/exp098-selector-rank-slot-features-on-exp073-train`
- Version: v1
- Status: complete
- Output: `experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/train_v1`
- Runtime: 14,959.909 sec

## 結果

| model | pooled OOF RMSE |
| --- | ---: |
| `lgb1` | 9.358151052 |
| `lgb2` | 9.366698537 |
| `lgb_mean` | 9.427447987 |
| `lgb0` | 9.732275226 |

`lgb1` が最良。exp073 raw anchor 9.526374749 から -0.168223697、exp077 policy 9.470514801 から -0.112363749 改善した。一方で exp092 best `lgb1` 9.322479896 より +0.035671157 悪い。

`lgb_mean` は 9.427447987 で exp073 / exp077 より改善するが、exp092 `lgb_mean` 9.343064066 より +0.084383921 悪い。

## Rank Slot 分布

| slot | pf_ancc | beam_mean | likpf_mean | sc_ens | hyb |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank1 | 0.336495 | 0.245513 | 0.417991 | 0.000000 | 0.000000 |
| rank2 | 0.250902 | 0.231920 | 0.517178 | 0.000000 | 0.000000 |
| rank3 | 0.412595 | 0.522564 | 0.064831 | 0.000002 | 0.000008 |

exp093 の現行 score では `pf_ancc` が top1 にならなかったが、この実装では `pf_ancc` が rank1 の 33.65% に入っている。`sc_ens` と `hyb` はほぼ rank slot に入っていない。

## Bucket

`lgb_mean` の distance bucket:

| bucket | rows | RMSE |
| --- | ---: | ---: |
| 000_050 | 38,650 | 1.182402 |
| 050_100 | 38,650 | 1.469994 |
| 100_250 | 115,950 | 2.275114 |
| 250_500 | 193,157 | 3.537170 |
| 500_1000 | 385,911 | 5.226223 |
| 1000_plus | 3,011,671 | 10.349936 |

## Worst Wells

`lgb_mean` の worst wells:

| well | rows | RMSE | mean error |
| --- | ---: | ---: | ---: |
| 86454a6f | 7,964 | 55.862236 | -51.082836 |
| fb03ae90 | 6,431 | 43.981632 | 42.028553 |
| 1b1eba53 | 4,655 | 41.806999 | -37.877210 |
| 389ae58f | 6,463 | 39.287521 | -36.945164 |
| 91b301ce | 6,570 | 34.555744 | 24.288912 |

## 特徴量重要度

特徴量重要度は保存済み。

- `exp098_selector_rank_slot_features_on_exp073_feature_importance.csv`
- `exp098_selector_rank_slot_features_on_exp073_feature_importance_mean.csv`
- `exp098_selector_rank_slot_features_on_exp073_feature_importance_mean_top.png`

上位には `rank1_u_curvature`、`rank2_u_curvature`、`rank3_u_curvature`、`rank3_u_slope`、`rank2_u_slope`、`rank1_u_slope` など rank-slot U-space shape 系が入った。

## 再現性

- deterministic anchor: false
- kernel version: `kentookumura/exp098-selector-rank-slot-features-on-exp073-train` v1
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- model manifest SHA: `dbc19bd4844e187335e3c0806883ede994eaf7e119ed0be483cbbc05e8dcb33e`
- prediction gzip SHA: `8644545a1e135a59f462b6f255fd205c73c7ce5d89b0d7ac4e6aec5efa6a6a63`
- lgb1 prediction SHA: `6a2aaf8a085a2dccfe5c7a013f371bac66860c1300dc25900ae93325daae19ca`
- lgb_mean prediction SHA: `12657ee7d87dc8e1d31b3d5ab7e3818abf7bfe851b1d65cec94a3fb9538a0088`
- inference kernel version: `kentookumura/exp098-selector-rank-slot-features-on-exp073-infer` v1
- inference prediction SHA: `b39dbb2c98db1416c99e71f37dc9558283de83a58be6e6c9dbf10cba59e16c8b`
- submission SHA: `1d32582f3f5984eeb9dd0bc5798b12cdc2e7aa863e0334691028901f0325125f`

## Inference

- Kernel: `kentookumura/exp098-selector-rank-slot-features-on-exp073-infer`
- Version: v1
- Status: complete
- Output: `experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/inference_v1`
- Selected: `rank_slot_u_disagreement` / `gpu_repro_guard_dp_threads8` / `lgb1`
- Model count: 5 fold boosters
- Feature count: 260
- Test rows / submission rows: 14,151 / 14,151
- Fallback rows: 0
- Prediction min / max / mean / std: 11590.388671875 / 12240.130859375 / 11905.629384697306 / 279.315775048125
- submit-check: PASS
- competition submit: complete

## Public LB

| ref | Public LB | note |
| --- | ---: | --- |
| `53927490` | 8.441 | exp098 submission after user correction |

Kaggle submission descriptions are blank. `ref=53927479` / Public LB 8.350 was initially attributed to exp098 by timing, but the user corrected it as exp092. Therefore exp098 uses `ref=53927490` / Public LB 8.441.

Public LB 8.441 improves exp077 ML route anchor 8.611 by -0.170 and exp073 raw anchor 8.780 by -0.339. It does not beat the ensemble route anchor exp082 7.601.

This means the rank-slot idea is useful: even after the 8.350 attribution correction, exp098 is not a failed direction. It is weaker than exp092 as a standalone ML route candidate, but it produced a real gain over the prior exp077 submitted/postprocessed anchor.

## 解釈

rank-slot structured features は train-side OOF では exp073 / exp077 を上回ったが、同じ exp073 派生の exp092 U-projection correction/disagreement には届かなかった。`lgb1` 単体が `lgb_mean` よりかなり強く、ensemble 平均で `lgb0` が足を引っ張っている。

exp092 とのマージでさらに伸びる可能性はある。exp092 は U-projection correction / disagreement を直接特徴量化しており、exp098 は PF/Beam/likelihood-PF 候補の rank、score、source、U-space shape を構造化しているため、完全に同じ信号ではない。ただしどちらも exp073/exp072 surface と U-space 情報に依存するので、全 64 rank-slot features をそのまま足すより、compact / top-n 版を exp092 に add-only で入れて、OOF、worst-well、near-row、path continuity を確認するのが妥当。

ユーザー依頼により `lgb1` 単体で inference と submit まで実行した。train-side OOF は exp092 より弱く、Public LB も exp092 8.350 には届かなかったが、8.441 で exp077 8.611 は更新した。一方で ensemble route の exp082 7.601 には届かない。

## 次

exp098 は exp077 を上回る有用な rank-slot 比較基準として保持する。ML route submitted anchor は exp092 8.350 に更新済みだが、follow-up は `compact_rank_slot_features_on_exp098` / `selector_topn_candidate_only_features` に加えて、exp092 feature surface へ compact rank-slot signals を add-only でマージする候補を優先的に検討する。
