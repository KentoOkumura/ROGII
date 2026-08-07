# exp073 特徴量一覧

exp073 GPU train v2 の保存済み LightGBM booster 15 個（3 configs x 5 folds）から、split count importance を再集計した特徴量一覧。
重要度は `LGBMRegressor.feature_importances_` の既定と同じ split count 基準で、`mean_split_importance` は 15 モデル平均。

## 集計元

- experiment: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- mode: `gpu_repro_guard_dp_threads8`
- train output: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2`
- feature source SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- rows / wells / features: 3,783,989 / 773 / 196
- boosters used: 15

## 一覧

| rank | feature | mean_split_importance | total_split_count | description |
|---:|---|---:|---:|---|
| 1 | `likpf_mean_d` | 5290.07 | 79351 | likelihood-weighted multi-seed PF の平均 TVT 推定 - last_known_tvt。 |
| 2 | `spatial_knn_dist` | 4512.33 | 67685 | formation surface KNN 補間で参照した近傍までの正規化距離。 |
| 3 | `frac` | 4363.80 | 65457 | 予測 tail 内の相対位置 0..1。 |
| 4 | `slp_b_d_50` | 4356.80 | 65352 | slp_50 を last_known_tvt から外挿した TVT - last_known_tvt。 |
| 5 | `dense_std` | 4297.47 | 64462 | dense ANCC imputer の近傍値から見た標準偏差。 |
| 6 | `dz` | 3955.93 | 59339 | 最後の既知点からの Z 差分。 |
| 7 | `tvt_dense50_d` | 3907.13 | 58607 | dense_ancc と late/prefix 後半 bias から作る TVT 推定 - last_known_tvt。 |
| 8 | `pf_z_delta` | 3883.13 | 58247 | pf_z - last_known_tvt。Z-aware PF 予測のアンカー差分。 |
| 9 | `dense_dist` | 3781.67 | 56725 | dense ANCC imputer の近傍距離。空間補間の遠さ。 |
| 10 | `dx` | 3770.87 | 56563 | 最後の既知点からの X 差分。 |
| 11 | `slp_b_d_all` | 3724.73 | 55871 | slp_all を last_known_tvt から外挿した TVT - last_known_tvt。 |
| 12 | `tvt_dense_d` | 3623.87 | 54358 | dense_ancc と prefix 全体 bias から作る TVT 推定 - last_known_tvt。 |
| 13 | `dy` | 3545.40 | 53181 | 最後の既知点からの Y 差分。 |
| 14 | `pf_vs_dense` | 3530.60 | 52959 | pf_ancc と dense TVT 候補の差。PF と dense surface の disagreement。 |
| 15 | `pf_ancc_delta` | 3494.00 | 52410 | pf_ancc - last_known_tvt。PF 予測のアンカーからの差分。 |
| 16 | `form_mean_d` | 3408.20 | 51123 | 6 formation surface TVT 候補の平均 - last_known_tvt。 |
| 17 | `spatial_vs_dense` | 3385.47 | 50782 | ANCC formation surface TVT 候補と dense TVT 候補の差。 |
| 18 | `pf_vs_z` | 3335.20 | 50028 | ANCC 型 PF と Z-aware PF の推定 TVT 差。2 種類の PF の disagreement。 |
| 19 | `tvt_densew_d` | 3324.00 | 49860 | dense_ancc と weighted least squares bias から作る TVT 推定 - last_known_tvt。 |
| 20 | `pf_vs_spatial` | 2922.93 | 43844 | pf_ancc と ANCC formation surface TVT 候補の差。 |
| 21 | `beam_vs_spatial` | 2825.87 | 42388 | conservative beam TVT と ANCC formation surface TVT 候補の差。 |
| 22 | `beam_vloose_d` | 2712.87 | 40693 | very loose beam search による TVT パス - last_known_tvt。 |
| 23 | `form_std_d` | 2689.93 | 40349 | 6 formation surface TVT 候補の標準偏差。formation surface 間の不一致。 |
| 24 | `dense_nb_std` | 2681.20 | 40218 | known prefix 近傍での dense ANCC 補間標準偏差平均。 |
| 25 | `form_rng_d` | 2568.67 | 38530 | 6 formation surface TVT 候補の最大値 - 最小値。formation surface のレンジ。 |
| 26 | `md_since` | 2567.40 | 38511 | 最後の既知点からの MD 距離。 |
| 27 | `beam_std_d` | 2515.27 | 37729 | 7 種類の beam search TVT パスの標準偏差。beam search の不確実性。 |
| 28 | `eval_len` | 2507.93 | 37619 | TVT_input が欠損していて予測対象となる tail 行数。 |
| 29 | `beam_vcons_d` | 2491.13 | 37367 | very conservative beam search による TVT パス - last_known_tvt。 |
| 30 | `slp_z` | 2450.67 | 36760 | known prefix の Z に対する TVT robust slope。 |
| 31 | `slp_50` | 2444.20 | 36663 | known prefix 末尾 50 点の MD に対する TVT robust slope。 |
| 32 | `dxy` | 2372.80 | 35592 | 最後の既知点からの XY 平面距離。 |
| 33 | `dense_ancc` | 2367.93 | 35519 | dense ANCC imputer による空間補間 ANCC 値。 |
| 34 | `spatial_ancc_d` | 2325.60 | 34884 | 空間 KNN で補間した ANCC surface 値と、anchor TVT 位置の typewell GR の差。 |
| 35 | `z` | 2311.87 | 34678 | 評価点の Z 座標。 |
| 36 | `pfx_rmse` | 2254.53 | 33818 | known prefix GR と typewell GR の対応 RMSE。prefix/typewell 一致度。 |
| 37 | `beam_stiff_d` | 2193.60 | 32904 | stiff beam-search setting による TVT パス - last_known_tvt。 |
| 38 | `known_len` | 2168.07 | 32521 | TVT_input が既知の prefix 行数。 |
| 39 | `cal_a` | 2158.93 | 32384 | known prefix の GR と typewell GR を合わせる affine calibration の slope。 |
| 40 | `tw_gr_mean` | 2138.27 | 32074 | typewell GR の平均。 |
| 41 | `cal_b` | 2107.20 | 31608 | known prefix の GR と typewell GR を合わせる affine calibration の intercept。 |
| 42 | `beam_mean_d` | 2103.07 | 31546 | 7 種類の beam search TVT パスの平均 - last_known_tvt。 |
| 43 | `beam_loose_d` | 2042.53 | 30638 | loose beam search による TVT パス - last_known_tvt。 |
| 44 | `slp_all` | 2040.80 | 30612 | known prefix 全体の MD に対する TVT robust slope。 |
| 45 | `dxdmd` | 2039.73 | 30596 | 局所的な dX/dMD。軌跡の X 方向勾配。 |
| 46 | `beam_cons_d` | 1941.07 | 29116 | conservative beam search による TVT パス - last_known_tvt。 |
| 47 | `pf_ancc_std` | 1928.07 | 28921 | ANCC 型 Particle Filter の粒子分布から見た TVT 標準偏差。PF の不確実性。 |
| 48 | `beam_mid_d` | 1852.53 | 27788 | middle beam-search setting による TVT パス - last_known_tvt。 |
| 49 | `beam_sm5_d` | 1792.07 | 26881 | beam search using GR smoothing radius 5 による TVT パス - last_known_tvt。 |
| 50 | `frac2` | 1762.13 | 26432 | frac の二乗。tail 内位置の非線形表現。 |
| 51 | `grs101` | 1652.47 | 24787 | GR の centered rolling standard deviation。window=101。 |
| 52 | `tw_range` | 1621.80 | 24327 | typewell TVT の範囲。 |
| 53 | `ktvt_range` | 1542.07 | 23131 | known prefix の TVT_input 範囲。 |
| 54 | `ktvt_std` | 1462.53 | 21938 | known prefix の TVT_input 標準偏差。 |
| 55 | `dydmd` | 1454.87 | 21823 | 局所的な dY/dMD。軌跡の Y 方向勾配。 |
| 56 | `grm101` | 1429.33 | 21440 | GR の centered rolling mean。window=101。 |
| 57 | `frm_rmse_ANCC` | 1420.73 | 21311 | ANCC surface + bias が known prefix TVT を再現する RMSE。 |
| 58 | `dzdmd` | 1343.87 | 20158 | 局所的な dZ/dMD。軌跡の上下方向勾配。 |
| 59 | `beam_med_d` | 1338.87 | 20083 | 7 種類の beam search TVT パスの中央値 - last_known_tvt。 |
| 60 | `frm_rmse_ASTNU` | 1246.47 | 18697 | ASTNU surface + bias が known prefix TVT を再現する RMSE。 |
| 61 | `pf_z` | 1022.87 | 15343 | Z 方向の軌跡変化も使う Particle Filter による絶対 TVT 推定値。 |
| 62 | `frm_rmse_ASTNL` | 1009.13 | 15137 | ASTNL surface + bias が known prefix TVT を再現する RMSE。 |
| 63 | `frm_rmse_EGFDU` | 984.20 | 14763 | EGFDU surface + bias が known prefix TVT を再現する RMSE。 |
| 64 | `pf_ancc` | 946.87 | 14203 | ANCC 型 Particle Filter による絶対 TVT 推定値。MD/Z/GR と typewell GR を使う。 |
| 65 | `tda80` | 876.87 | 13153 | 評価点 GR と、last_known_tvt +80 位置の typewell GR の差。 |
| 66 | `sqrt_frac` | 867.93 | 13019 | frac の平方根。tail 初期側を広げた位置表現。 |
| 67 | `frm_rmse_EGFDL` | 848.73 | 12731 | EGFDL surface + bias が known prefix TVT を再現する RMSE。 |
| 68 | `grs51` | 833.80 | 12507 | GR の centered rolling standard deviation。window=51。 |
| 69 | `frm_rmse_BUDA` | 760.87 | 11413 | BUDA surface + bias が known prefix TVT を再現する RMSE。 |
| 70 | `tvtF_ASTNU` | 676.27 | 10144 | ASTNU formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 71 | `grm51` | 652.60 | 9789 | GR の centered rolling mean。window=51。 |
| 72 | `last_known_tvt` | 642.93 | 9644 | 予測対象区間直前の最後の既知 TVT_input。モデルの基準値で、target は TVT - last_known_tvt。 |
| 73 | `tvtF_ANCC` | 629.33 | 9440 | ANCC formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 74 | `tvtFw_ASTNU` | 576.80 | 8652 | ASTNU formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 75 | `tdbc40` | 560.40 | 8406 | 評価点 GR と、beam_ref +40 位置の typewell GR の差。 |
| 76 | `gr_env` | 536.80 | 8052 | GR の rolling 21 max。局所 envelope。 |
| 77 | `tvtF50_ASTNU` | 532.80 | 7992 | ASTNU formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 78 | `tvtF_EGFDL` | 529.00 | 7935 | EGFDL formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 79 | `tvtF_ASTNL` | 516.87 | 7753 | ASTNL formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 80 | `tdpf30` | 508.07 | 7621 | 評価点 GR と、PF ANCC +30 位置の typewell GR の差。 |
| 81 | `tda40` | 501.67 | 7525 | 評価点 GR と、last_known_tvt +40 位置の typewell GR の差。 |
| 82 | `tvtF_EGFDU` | 494.87 | 7423 | EGFDU formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 83 | `tvtFw_ANCC` | 489.60 | 7344 | ANCC formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 84 | `tvtF_BUDA` | 483.33 | 7250 | BUDA formation surface KNN と prefix 全体 bias から作る TVT 推定値。 |
| 85 | `dense_rmse` | 467.53 | 7013 | known prefix 上で dense ANCC 補正が TVT を再現する RMSE。 |
| 86 | `tvtF50_ANCC` | 434.80 | 6522 | ANCC formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 87 | `glead30` | 431.93 | 6479 | 評価点より 30 行後の GR。推論時にも水平井ログ内で見える GR lead。 |
| 88 | `tdpf15` | 417.53 | 6263 | 評価点 GR と、PF ANCC +15 位置の typewell GR の差。 |
| 89 | `tvtFw_EGFDL` | 417.07 | 6256 | EGFDL formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 90 | `tdpf-30` | 406.73 | 6101 | 評価点 GR と、PF ANCC -30 位置の typewell GR の差。 |
| 91 | `tdpf-2` | 406.67 | 6100 | 評価点 GR と、PF ANCC -2 位置の typewell GR の差。 |
| 92 | `tvtFw_ASTNL` | 402.33 | 6035 | ASTNL formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 93 | `tvtFw_BUDA` | 387.93 | 5819 | BUDA formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 94 | `tvtFw_EGFDU` | 387.27 | 5809 | EGFDU formation surface KNN と weighted least squares bias から作る TVT 推定値。 |
| 95 | `bw_early_ASTNU` | 383.27 | 5749 | ASTNU surface で prefix early segment から推定した bias。 |
| 96 | `tdpf-4` | 380.47 | 5707 | 評価点 GR と、PF ANCC -4 位置の typewell GR の差。 |
| 97 | `tvtF50_ASTNL` | 377.47 | 5662 | ASTNL formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 98 | `tvtF50_BUDA` | 377.33 | 5660 | BUDA formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 99 | `tvtF50_EGFDL` | 377.27 | 5659 | EGFDL formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 100 | `glag30` | 376.53 | 5648 | 評価点より 30 行前の GR。 |
| 101 | `tdpf-15` | 373.80 | 5607 | 評価点 GR と、PF ANCC -15 位置の typewell GR の差。 |
| 102 | `bw_early_ANCC` | 372.93 | 5594 | ANCC surface で prefix early segment から推定した bias。 |
| 103 | `tvtF50_EGFDU` | 362.07 | 5431 | EGFDU formation surface KNN と prefix 後半/late bias から作る TVT 推定値。 |
| 104 | `tdpf8` | 361.93 | 5429 | 評価点 GR と、PF ANCC +8 位置の typewell GR の差。 |
| 105 | `tda-40` | 360.73 | 5411 | 評価点 GR と、last_known_tvt -40 位置の typewell GR の差。 |
| 106 | `tda-80` | 359.73 | 5396 | 評価点 GR と、last_known_tvt -80 位置の typewell GR の差。 |
| 107 | `grs21` | 352.07 | 5281 | GR の centered rolling standard deviation。window=21。 |
| 108 | `tdpf4` | 351.20 | 5268 | 評価点 GR と、PF ANCC +4 位置の typewell GR の差。 |
| 109 | `tdpf-8` | 339.67 | 5095 | 評価点 GR と、PF ANCC -8 位置の typewell GR の差。 |
| 110 | `sc25_sc` | 337.47 | 5062 | window 25 の normalized cross-correlation score。大きいほど GR パターン一致が強い。 |
| 111 | `tdbc-20` | 336.67 | 5050 | 評価点 GR と、beam_ref -20 位置の typewell GR の差。 |
| 112 | `tdpf2` | 333.53 | 5003 | 評価点 GR と、PF ANCC +2 位置の typewell GR の差。 |
| 113 | `bw_ANCC` | 332.53 | 4988 | ANCC surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 114 | `tdbc-40` | 317.13 | 4757 | 評価点 GR と、beam_ref -40 位置の typewell GR の差。 |
| 115 | `tdbc20` | 293.73 | 4406 | 評価点 GR と、beam_ref +20 位置の typewell GR の差。 |
| 116 | `bw_ASTNU` | 288.20 | 4323 | ASTNU surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 117 | `sc25_d` | 283.20 | 4248 | window 25 の normalized cross-correlation TVT 推定 - last_known_tvt。 |
| 118 | `bw_early_EGFDU` | 281.47 | 4222 | EGFDU surface で prefix early segment から推定した bias。 |
| 119 | `bww_ANCC` | 278.07 | 4171 | ANCC surface で weighted least squares により推定した bias。 |
| 120 | `tdbc5` | 276.60 | 4149 | 評価点 GR と、beam_ref +5 位置の typewell GR の差。 |
| 121 | `bww_ASTNU` | 270.47 | 4057 | ASTNU surface で weighted least squares により推定した bias。 |
| 122 | `tda20` | 268.40 | 4026 | 評価点 GR と、last_known_tvt +20 位置の typewell GR の差。 |
| 123 | `tdbc-3` | 268.20 | 4023 | 評価点 GR と、beam_ref -3 位置の typewell GR の差。 |
| 124 | `bw_early_ASTNL` | 266.07 | 3991 | ASTNL surface で prefix early segment から推定した bias。 |
| 125 | `tdbc10` | 265.80 | 3987 | 評価点 GR と、beam_ref +10 位置の typewell GR の差。 |
| 126 | `tda-20` | 259.80 | 3897 | 評価点 GR と、last_known_tvt -20 位置の typewell GR の差。 |
| 127 | `tda-10` | 258.80 | 3882 | 評価点 GR と、last_known_tvt -10 位置の typewell GR の差。 |
| 128 | `tdbc-10` | 251.40 | 3771 | 評価点 GR と、beam_ref -10 位置の typewell GR の差。 |
| 129 | `tda10` | 250.87 | 3763 | 評価点 GR と、last_known_tvt +10 位置の typewell GR の差。 |
| 130 | `tda5` | 244.80 | 3672 | 評価点 GR と、last_known_tvt +5 位置の typewell GR の差。 |
| 131 | `tdpf0` | 244.13 | 3662 | 評価点 GR と、PF ANCC +0 位置の typewell GR の差。 |
| 132 | `tdbc3` | 243.93 | 3659 | 評価点 GR と、beam_ref +3 位置の typewell GR の差。 |
| 133 | `tdbc-5` | 243.67 | 3655 | 評価点 GR と、beam_ref -5 位置の typewell GR の差。 |
| 134 | `glead15` | 242.93 | 3644 | 評価点より 15 行後の GR。推論時にも水平井ログ内で見える GR lead。 |
| 135 | `gr_nrg` | 240.87 | 3613 | GR^2 の rolling 21 mean の平方根。局所エネルギー。 |
| 136 | `gr_vs_slp_all` | 235.87 | 3538 | 評価点 GR と slp_all 外挿 TVT 位置の typewell GR の差。 |
| 137 | `tdbc0` | 232.93 | 3494 | 評価点 GR と、beam_ref +0 位置の typewell GR の差。 |
| 138 | `dense_bias` | 231.93 | 3479 | known prefix 上の dense ANCC residual bias 平均。 |
| 139 | `glag15` | 223.73 | 3356 | 評価点より 15 行前の GR。 |
| 140 | `bw_mid_ASTNU` | 217.47 | 3262 | ASTNU surface で prefix mid segment から推定した bias。 |
| 141 | `bw_early_EGFDL` | 216.80 | 3252 | EGFDL surface で prefix early segment から推定した bias。 |
| 142 | `bw_mid_ANCC` | 206.47 | 3097 | ANCC surface で prefix mid segment から推定した bias。 |
| 143 | `tda-5` | 204.13 | 3062 | 評価点 GR と、last_known_tvt -5 位置の typewell GR の差。 |
| 144 | `bw50_ASTNU` | 201.27 | 3019 | ASTNU surface で prefix 後半/late segment から推定した bias。 |
| 145 | `grm21` | 199.13 | 2987 | GR の centered rolling mean。window=21。 |
| 146 | `bw50_ANCC` | 198.93 | 2984 | ANCC surface で prefix 後半/late segment から推定した bias。 |
| 147 | `bww_ASTNL` | 187.20 | 2808 | ASTNL surface で weighted least squares により推定した bias。 |
| 148 | `bw_ASTNL` | 184.80 | 2772 | ASTNL surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 149 | `bww_EGFDU` | 183.27 | 2749 | EGFDU surface で weighted least squares により推定した bias。 |
| 150 | `bw_EGFDU` | 179.53 | 2693 | EGFDU surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 151 | `gr_vs_tw_anc` | 174.67 | 2620 | 評価点 GR と last_known_tvt 位置の typewell GR の差。 |
| 152 | `bw_early_BUDA` | 168.00 | 2520 | BUDA surface で prefix early segment から推定した bias。 |
| 153 | `bw_mid_ASTNL` | 149.80 | 2247 | ASTNL surface で prefix mid segment から推定した bias。 |
| 154 | `bw50_ASTNL` | 144.87 | 2173 | ASTNL surface で prefix 後半/late segment から推定した bias。 |
| 155 | `bw_mid_EGFDU` | 144.33 | 2165 | EGFDU surface で prefix mid segment から推定した bias。 |
| 156 | `bww_EGFDL` | 141.20 | 2118 | EGFDL surface で weighted least squares により推定した bias。 |
| 157 | `bw50_EGFDU` | 135.33 | 2030 | EGFDU surface で prefix 後半/late segment から推定した bias。 |
| 158 | `bw_mid_EGFDL` | 133.60 | 2004 | EGFDL surface で prefix mid segment から推定した bias。 |
| 159 | `bw_EGFDL` | 133.00 | 1995 | EGFDL surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 160 | `sc15_sc` | 117.33 | 1760 | window 15 の normalized cross-correlation score。大きいほど GR パターン一致が強い。 |
| 161 | `bw_BUDA` | 114.47 | 1717 | BUDA surface で known prefix 全体から推定した TVT+Z-surface bias。 |
| 162 | `bww_BUDA` | 108.93 | 1634 | BUDA surface で weighted least squares により推定した bias。 |
| 163 | `bw_mid_BUDA` | 108.13 | 1622 | BUDA surface で prefix mid segment から推定した bias。 |
| 164 | `bw50_EGFDL` | 105.67 | 1585 | EGFDL surface で prefix 後半/late segment から推定した bias。 |
| 165 | `glag5` | 92.40 | 1386 | 評価点より 5 行前の GR。 |
| 166 | `tda0` | 86.47 | 1297 | 評価点 GR と、last_known_tvt +0 位置の typewell GR の差。 |
| 167 | `glead5` | 83.20 | 1248 | 評価点より 5 行後の GR。推論時にも水平井ログ内で見える GR lead。 |
| 168 | `bw50_BUDA` | 80.93 | 1214 | BUDA surface で prefix 後半/late segment から推定した bias。 |
| 169 | `grm5` | 79.80 | 1197 | GR の centered rolling mean。window=5。 |
| 170 | `sc15_d` | 73.40 | 1101 | window 15 の normalized cross-correlation TVT 推定 - last_known_tvt。 |
| 171 | `gr` | 49.20 | 738 | 評価点の GR。欠損は補間済み。 |
| 172 | `sig_std` | 43.40 | 651 | PF、beam、SC、formation ANCC、dense TVT 候補群の行方向標準偏差。候補間 disagreement。 |
| 173 | `sc8_sc` | 43.13 | 647 | window 8 の normalized cross-correlation score。大きいほど GR パターン一致が強い。 |
| 174 | `grs5` | 41.60 | 624 | GR の centered rolling standard deviation。window=5。 |
| 175 | `sig_mean_d` | 39.60 | 594 | PF、beam、SC、formation ANCC、dense TVT 候補群の平均 - last_known_tvt。 |
| 176 | `glag1` | 27.53 | 413 | 評価点より 1 行前の GR。 |
| 177 | `glead1` | 21.20 | 318 | 評価点より 1 行後の GR。推論時にも水平井ログ内で見える GR lead。 |
| 178 | `sc_cons_d` | 14.67 | 220 | SC 8/15/25 の平均 TVT 推定 - last_known_tvt。 |
| 179 | `gr_d1` | 13.00 | 195 | GR の一次差分。 |
| 180 | `hyb_d` | 12.47 | 187 | beam_ref と SC ensemble を sc_trust で blend した TVT 推定 - last_known_tvt。 |
| 181 | `sc8_d` | 10.73 | 161 | window 8 の normalized cross-correlation TVT 推定 - last_known_tvt。 |
| 182 | `sc_vs_beam` | 10.67 | 160 | SC ensemble TVT と conservative beam TVT の差。 |
| 183 | `sc_ens_d` | 7.87 | 118 | multi-scale NCC ensemble による TVT 推定 - last_known_tvt。 |
| 184 | `tdsc-30` | 7.73 | 116 | 評価点 GR と、SC ensemble -30 位置の typewell GR の差。 |
| 185 | `tdsc8` | 7.73 | 116 | 評価点 GR と、SC ensemble +8 位置の typewell GR の差。 |
| 186 | `tdsc-2` | 7.27 | 109 | 評価点 GR と、SC ensemble -2 位置の typewell GR の差。 |
| 187 | `tdsc30` | 7.20 | 108 | 評価点 GR と、SC ensemble +30 位置の typewell GR の差。 |
| 188 | `tdsc15` | 7.07 | 106 | 評価点 GR と、SC ensemble +15 位置の typewell GR の差。 |
| 189 | `tdsc-8` | 7.00 | 105 | 評価点 GR と、SC ensemble -8 位置の typewell GR の差。 |
| 190 | `tdsc-15` | 6.20 | 93 | 評価点 GR と、SC ensemble -15 位置の typewell GR の差。 |
| 191 | `tdsc2` | 5.93 | 89 | 評価点 GR と、SC ensemble +2 位置の typewell GR の差。 |
| 192 | `tdsc4` | 5.73 | 86 | 評価点 GR と、SC ensemble +4 位置の typewell GR の差。 |
| 193 | `tdsc0` | 5.60 | 84 | 評価点 GR と、SC ensemble +0 位置の typewell GR の差。 |
| 194 | `tdsc-4` | 5.00 | 75 | 評価点 GR と、SC ensemble -4 位置の typewell GR の差。 |
| 195 | `gr_d2` | 1.53 | 23 | GR の二次差分。 |
| 196 | `sc_trust` | 0.00 | 0 | 既知 prefix 長から決めた SC 信頼度。len(known)/200 を 0.6 上限で clip。 |

## メモ

- `*_d` は原則として `last_known_tvt` からの差分。
- `tda*` / `tdbc*` / `tdsc*` / `tdpf*` は、評価点 GR と typewell GR の差を、anchor / beam / SC / PF の各 TVT 候補に offset を足した位置で見た特徴量。
- `tvtF*` / `bw*` / `frm_rmse*` は training-only formation columns を直接使わず、train wells 由来の空間 KNN surface と known prefix bias から作った fold-safe な補助特徴量。
