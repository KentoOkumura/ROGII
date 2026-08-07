# exp083_pf_beam_true_tvt_2d_well_eda 結果

## 仮説

exp073 の入力 feature cache である exp072 に含まれる PF/Beam 系候補を true TVT と well ごとに重ねると、PF/Beam が効く / 外す条件を集計 RMSE より直接確認できる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- Anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 検証: visual EDA on pseudo-hidden train tails
- メトリック: RMSE
- シード: 42

## 結果

Kaggle train v2 completed for the full candidate view. Kaggle train v3 completed for the representative clean view requested after reviewing readability. Kaggle train v4 completed for the clean all-well view. Kaggle train v7 completed for the ANCC/Z physical decomposition view. Kaggle train v11 completed for the all-well diagnostic view with PF Z, formation bands, Z background, and a lower dZ/dMD panel. Kaggle train v14 completed for the selected-well known-prefix replay overlay view. The separate v12 ML OOF + known raw TVT probe notebook completed for all 773 wells.

| 項目 | 値 |
| --- | --- |
| rows | 3,783,989 |
| wells | 773 |
| v2 full candidate plots | 70 |
| v3 clean representative plots | 70 |
| v4 clean all-well plots | 773 |
| v7 ANCC/Z decomposition plots | 773 |
| v11 current diagnostic plots | 773 |
| v14 known-prefix replay overlay plots | 6 |
| v14 known-prefix replay rows | 1,536 |
| v12 ML OOF + known raw TVT probe plots | 773 |
| v12 ML OOF coverage | 1.0 |
| v12 raw TVT probe coverage | 1.0 |
| primary PF mean well RMSE | 10.844610 |
| primary Beam mean well RMSE | 12.587074 |
| anchor mean well RMSE | 12.812479 |

生成物:

- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plot_manifest.csv`
- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots.zip`
- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots/`
- `kaggle/output/train_v14_known_prefix_overlay/artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plot_manifest.csv`
- `kaggle/output/train_v14_known_prefix_overlay/artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plots/`
- `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plot_manifest.csv`
- `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots.zip`
- `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots/`

## 再現性

- deterministic anchor: false
- seed policy: no RNG except stable representative sampling
- source SHA: exp072 v2 gzip SHA `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- decompressed content SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- model / submission SHA: 対象外

## 解釈

well 平均では PF ANCC が anchor / Beam mean より良い。ただし worst PF wells では PF が大きく外れ、Beam / anchor が勝つ例もあるため、直接 PF 置換ではなく disagreement/error map と confidence-based clip の材料にする。

v3/v4 clean plot では暴れていた `sc_ens` / `hyb` / formation / dense candidates を除外し、5 本だけを表示する。PF ANCC は青、Beam mean はオレンジ、likelihood-PF は緑、anchor はグレー破線。v4 は `all_wells__{well_id}.png` として全 773 well を出力した。

v11 current diagnostic plot では、true TVT の急変要因を読むため、clean plot の `true TVT`、`PF ANCC`、`PF Z`、`Beam mean`、`Likelihood PF mean`、`last anchor` を維持したうえで、背景に raw train horizontal CSV の `Z` と `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` の formation band を薄く表示する。下段には `dZ/dMD` を表示する。raw formation columns は train-only なので EDA 専用であり、推論特徴や提出ルールには直接使わない。

v14 known-prefix replay overlay では、selected 6 wells について exp083 v11 系の plot に known `TVT_input` prefix を prediction start 前へ追加し、exp169 と同じ known prefix 末尾 256 rows holdout 条件で PF/Beam を再生成した。表示上は通常の exp072 tail candidate を実線、known prefix replay を `known replay PF ANCC` / `known replay PF Z` / `known replay Beam mean` / `known replay Likelihood PF mean` の破線として重ねている。exp169 本体の保存 output は offset summary が中心で row-level prefix trajectory は保存していなかったため、v14 では exp083 側で同条件の replay を再実行している。

v12 ML OOF + known raw TVT probe 別 notebook では、全 773 well の exp072 feature-cache rows に exp148 `lgb_mean` OOF と raw train `TVT` probe を重ねた。`TVT_input` の known prefix rows は plot frame に追加していない。Kaggle v1 logs では PF/Beam rows 3,783,989、exp148 OOF rows 3,783,989、joined rows 3,783,989、ML OOF coverage 1.0、raw TVT probe coverage 1.0、`tvt_input_prefix_plotted=false` を確認した。zip 内 PNG 773 件の integrity は PASS。

v11 は全 773 well の plot zip を生成した。ローカルでは旧plot成果物を削除したうえで v11 manifest と plots zip を取得し、zip 内 773 PNG の整合性を確認した。Kaggle CLI の個別PNG取得は遅かったため、ローカルの展開済みPNGディレクトリは検証済みzipから再作成した。

v11 の代表 plot を確認すると、急変が目立つ `91b301ce`、`ba48188d`、`fef8af96` では true TVT の大きな下降と `-Z scaled` の動き、下段の正の `dZ/dMD` が同時に出ており、主因は広域 formation band の境界ではなく坑跡 Z 成分の寄与と読むのが自然。`43e16325` のように true TVT が上昇して PF Z / Likelihood PF とよく同期する例もあり、これはPF側がZ由来のTVT形状を捉えている。一方で `1b1eba53` は ANCC coverage が 0 で、PF/Beam/anchor が true TVT の層準上昇を追えていないため、formation band の可視化だけでは説明しにくい局所相関ずれ候補として扱う。

## 次

v11 plot を目視し、TVT急変が `dZ/dMD`、formation band、または `PF Z` のどれとも同期しない well を代表例として分類する。同期しない場合は、粗いZ/formation面ではなく、true TVTラベル側の局所的な相関先ジャンプや層内位置の切り替わりを疑う。
