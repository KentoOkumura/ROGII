# exp083_pf_beam_true_tvt_2d_well_eda

## 状態

- ルート: `pf_beam`
- 状態: `eda_completed`
- CV: -
- Public LB: -
- Submit: なし
- 作成日: 2026-06-19
- 親実験: `exp072_exp063_full_replay_feature_cache`
- Anchor: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`

## 仮説

PF/Beam が効く well と外す well は、RMSE 集計だけでは形状の違いが見えにくい。現 anchor exp073 の入力 feature cache である exp072 から、true TVT と PF/Beam 系候補を well ごとに重ねて見ることで、次の disagreement/error map 実験で読むべき条件を整理する。

## 実装

- exp072 の full replay train cache を読む。
- `true_tvt = last_known_tvt + target` に戻す。
- `beam_*_d`、`likpf_mean_d`、`hyb_d` などは `last_known_tvt + delta` に戻す。
- well-level RMSE、不一致、confidence/context を CSV 保存する。
- 代表 well または全 well の PNG と plot manifest を保存する。

## 検証方針

- `validate_experiment.py` で構成と TODO 不在を確認する。
- exp072 型の合成 CSV で `target` delta と `*_d` 候補の TVT 空間復元、well summary、metrics 出力を確認する。
- PNG 作成と exp072 kernel source mount は Kaggle train notebook で確認する。

## 所見

Kaggle train v2 で full candidate plot を完了。v3/v4 で `true TVT / PF ANCC / Beam mean / Likelihood PF mean / last anchor` の clean plot を作成した。v11 では clean all-well plot に `PF Z`、全 formation band、`Z` 背景、下段 `dZ/dMD` を追加した。v14 では selected 6 wells について known `TVT_input` prefix を prediction start 前に追加し、exp169 と同じ prefix holdout replay 条件で再生成した PF/Beam を破線 overlay した。exp072 feature cache 3,783,989 rows / 773 wells を読んだ。

- primary PF mean well RMSE: 10.844610
- primary Beam mean well RMSE: 12.587074
- anchor mean well RMSE: 12.812479
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- current v14 overlay output: `kaggle/output/train_v14_known_prefix_overlay/`
- current v14 plot dir: `kaggle/output/train_v14_known_prefix_overlay/artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plots/`
- current v14 manifest: `kaggle/output/train_v14_known_prefix_overlay/artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plot_manifest.csv`
- v11 all-well plot zip remains: `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots.zip`

## 実行入口

- EDA notebook: `exp083_pf_beam_true_tvt_2d_well_eda_train.ipynb`
- Inference notebook: no-op。提出ファイルは作らない。
- Kaggle 準備:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
```

## 注意

true TVT は EDA と評価だけに使う。ここで見えた数例を根拠に、直接 hard router や置換ルールへ進めない。
