# 特徴量重要度順の説明 2026-06-24

## 目的

OOF 分析で参照した特徴量重要度を、人間が読める説明表として `docs/analysis` に置く。CSV は置かず、重要度順、特徴量の意味、誤差分析上の読み方を Markdown に転記する。

## 参照元

- 全 196 特徴量の既存説明: `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/feature_list.md`
- OOF feature / error readout: `experiments/exp086_oof_feature_importance_error_readout/result.md`
- exp086 feature summary: `/tmp/kaggle-output/exp086_oof_feature_importance_error_readout/train_v1/artifacts/exp086_oof_feature_importance_error_readout_feature_summary.csv`

ここでの rank は exp086 の `gain_mean` 順。既存の `feature_list.md` は exp073 LightGBM 15 boosters の split count 順なので、順位は一致しない。

## 読み方

- `*_d`: 原則として `last_known_tvt` からの差分。
- `pf_*`: Particle Filter 系。
- `beam_*`: Beam search 系。
- `tvt_dense*` / `dense_*`: dense ANCC / dense surface 補間系。
- `form_*` / `spatial_*`: formation surface / 空間 KNN surface 系。
- `slp_*`: known prefix から推定した傾き。
- `*_vs_*`: 候補同士の disagreement。誤差・不確実性 signal として重要。

## 重要度上位30

| rank | feature | category | 説明 | 誤差分析上の読み |
| ---: | --- | --- | --- | --- |
| 1 | `likpf_mean_d` | likelihood-PF | likelihood-weighted multi-seed PF の平均 TVT 推定から `last_known_tvt` を引いた差分。 | 単体候補として最重要。大きく正方向に振れる bucket は baseline RMSE 12.112 で error lift が大きい。 |
| 2 | `tvt_dense50_d` | dense surface | dense ANCC と prefix 後半 bias から作る TVT 推定差分。 | dense 系候補の高値 bucket は baseline RMSE 12.812。tail error の主要 signal。 |
| 3 | `tvt_densew_d` | dense surface | dense ANCC と weighted least squares bias から作る TVT 推定差分。 | high bucket の baseline RMSE 12.852、MAE lift +2.534。dense 系では特に外れ regime をよく示す。 |
| 4 | `pf_ancc_delta` | PF | `pf_ancc - last_known_tvt`。ANCC 型 PF 予測のアンカー差分。 | PF が大きく動く場面の signal。high bucket baseline RMSE 12.504。 |
| 5 | `tvt_dense_d` | dense surface | dense ANCC と prefix 全体 bias から作る TVT 推定差分。 | high bucket baseline RMSE 12.687。`tvt_dense50_d` / `tvt_densew_d` と同系統の tail-risk signal。 |
| 6 | `pf_z_delta` | PF | `pf_z - last_known_tvt`。Z-aware PF 予測のアンカー差分。 | 負方向に大きい bucket で baseline RMSE 12.192。ANCC 型 PF と違う失敗 regime を拾う。 |
| 7 | `form_mean_d` | formation surface | 6 formation surface TVT 候補の平均から `last_known_tvt` を引いた差分。 | high bucket baseline RMSE 12.664。formation surface が強く動く well は tail risk が高い。 |
| 8 | `slp_z` | slope | known prefix の Z に対する TVT robust slope。 | importance は高いが worst bucket lift は比較的小さい。軌跡と TVT 勾配の基礎特徴。 |
| 9 | `slp_50` | slope | known prefix 末尾 50 点の MD に対する TVT robust slope。 | prefix 末尾の局所傾き。外挿候補と組み合わさって効く。 |
| 10 | `dense_nb_std` | dense uncertainty | known prefix 近傍での dense ANCC 補間標準偏差平均。 | Spearman absolute-error correlation 0.079。dense surface の不確実性特徴。 |
| 11 | `dense_dist` | dense uncertainty | dense ANCC imputer の近傍距離。空間補間の遠さ。 | Spearman absolute-error correlation 0.119。遠い補間は error / uncertainty signal として強い。 |
| 12 | `pf_vs_dense` | disagreement | `pf_ancc` と dense TVT 候補の差。PF と dense surface の disagreement。 | worst bucket MAE lift +2.632 で最大級。confidence / gate / sample weight に優先して使う。 |
| 13 | `known_len` | prefix length | `TVT_input` が既知の prefix 行数。 | 長い prefix が必ず簡単とは限らない。well regime と tail 長の proxy。 |
| 14 | `slp_all` | slope | known prefix 全体の MD に対する TVT robust slope。 | `slp_50` よりグローバルな prefix 傾き。 |
| 15 | `beam_vs_spatial` | disagreement | conservative beam TVT と ANCC formation surface TVT 候補の差。 | 大きな負方向 bucket で baseline RMSE 12.742。Beam と spatial surface の不一致 signal。 |
| 16 | `pfx_rmse` | prefix fit | known prefix GR と typewell GR の対応 RMSE。prefix/typewell 一致度。 | prefix と typewell の対応が悪い well で uncertainty が増える。 |
| 17 | `beam_vloose_d` | beam candidate | very loose beam search による TVT path 差分。 | loose beam が負方向に大きく振れる bucket で baseline RMSE 12.367。候補集合の振れ幅 signal。 |
| 18 | `tw_gr_mean` | typewell log | typewell GR の平均。 | 地質 / typewell 側の baseline context。単独より相互作用で効く。 |
| 19 | `dx` | trajectory | 最後の既知点からの X 差分。 | spatial drift proxy。大きい bucket で baseline RMSE 12.498。 |
| 20 | `ktvt_range` | prefix TVT shape | known prefix の `TVT_input` 範囲。 | prefix 内 TVT 変動の大きさ。well の曲がり方 / scale を示す。 |
| 21 | `tw_range` | typewell shape | typewell TVT の範囲。 | typewell 側の TVT scale。地質 surface の広がり proxy。 |
| 22 | `pf_vs_z` | disagreement | ANCC 型 PF と Z-aware PF の推定 TVT 差。 | high bucket baseline RMSE 12.647。2種類の PF が割れる場面は不安定。 |
| 23 | `beam_stiff_d` | beam candidate | stiff beam-search setting による TVT path 差分。 | Beam family の conservative / loose 方向の比較材料。 |
| 24 | `ktvt_std` | prefix TVT shape | known prefix の `TVT_input` 標準偏差。 | prefix TVT の変動量。tail drift の間接 proxy。 |
| 25 | `eval_len` | tail length | `TVT_input` が欠損していて予測対象となる tail 行数。 | high bucket baseline RMSE 12.354。long tail が global RMSE を支配するため重要。 |
| 26 | `slp_b_d_50` | slope extrapolation | `slp_50` を `last_known_tvt` から外挿した TVT 差分。 | high bucket baseline RMSE 12.752。局所傾き外挿が大きく動く tail は危険。 |
| 27 | `dz` | trajectory | 最後の既知点からの Z 差分。 | high bucket baseline RMSE 11.604。vertical drift の基本特徴。 |
| 28 | `beam_std_d` | beam uncertainty | 7 種類の beam search TVT path の標準偏差。 | Spearman absolute-error correlation 0.175 で上位特徴中最強。uncertainty feature として最優先。 |
| 29 | `cal_a` | GR calibration | known prefix の GR と typewell GR を合わせる affine calibration の slope。 | GR scale mismatch の補正係数。地質ログ対応の良し悪しを示す。 |
| 30 | `cal_b` | GR calibration | known prefix の GR と typewell GR を合わせる affine calibration の intercept。 | GR offset mismatch の補正係数。`cal_a` と対で見る。 |

## 重要度上位の構造

上位は大きく 5 系統に分かれる。

| 系統 | 主な特徴量 | 読み |
| --- | --- | --- |
| PF / likelihood-PF 候補 | `likpf_mean_d`, `pf_ancc_delta`, `pf_z_delta` | 候補値そのものが強い。単体採用ではなく、壊れる regime を識別する必要がある。 |
| dense surface 候補 | `tvt_dense50_d`, `tvt_densew_d`, `tvt_dense_d`, `dense_dist`, `dense_nb_std` | tail error と強く結びつく。`likpf_mean` が外れる well で oracle になりやすいが、direct replacement は危険。 |
| disagreement | `pf_vs_dense`, `beam_vs_spatial`, `pf_vs_z` | 予測候補同士の不一致。confidence / gate / sample weight の入力として最重要。 |
| slope / trajectory | `slp_z`, `slp_50`, `slp_all`, `slp_b_d_50`, `dx`, `dz` | tail の幾何と prefix からの外挿方向を表す。long tail で効きやすい。 |
| uncertainty / length | `beam_std_d`, `dense_nb_std`, `dense_dist`, `known_len`, `eval_len` | 予測難度や外れやすさの proxy。特に `beam_std_d` は abs error 相関が強い。 |

## 実験への示唆

1. 次の confidence feature は `beam_std_d`、`dense_dist`、`dense_nb_std`、`pf_vs_dense`、`eval_len` を優先する。
2. `likpf_mean_d` は強いが、単体で外れる well があるため、`pf_vs_dense` / `beam_std_d` / `dense_dist` と組み合わせて gate する。
3. dense 系特徴は correction 候補として有望だが、high bucket が error lift そのものでもあるため、direct replacement ではなく low-switch gate に限定する。
4. `eval_len` と `slp_b_d_50` は tail 対策の基本特徴。near-prefix を壊さない fade-in / clipping と一緒に使う。
5. `cal_a` / `cal_b` / `pfx_rmse` は GR 対応の品質 signal。candidate selector / verifier の補助特徴として残す価値がある。

## 既存の全特徴量リスト

全 196 特徴量の説明は `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/feature_list.md` にある。上位30以外を確認する場合は同ファイルを正とする。
