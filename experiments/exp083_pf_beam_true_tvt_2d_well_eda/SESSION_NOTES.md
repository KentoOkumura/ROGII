# exp083_pf_beam_true_tvt_2d_well_eda セッションノート

## 目的

現 anchor `exp073_gpu_reproducibility_guard_for_exp063_full_replay` の入力である `exp072_exp063_full_replay_feature_cache` から、PF/Beam 系候補と true TVT を well ごとに 2D plot し、成功例・失敗例・高不一致例を形状で確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: `eda_completed`
- CV: なし
- LB: なし
- Submit: なし

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
uv run python scripts/new_experiment.py --name exp083_pf_beam_true_tvt_2d_well_eda --source templates/experiment
```

実装内容:

- `config.yaml` を親 `exp072`、anchor `exp073`、route `pf_beam` に更新。
- `pf_beam_true_tvt_eda.py` を追加。
- exp072 feature cache の `target` を `true_tvt = last_known_tvt + target` に戻す処理を追加。
- `beam_*_d`、`sc_ens_d`、`hyb_d`、`likpf_mean_d`、`tvt_dense_d` を TVT 空間へ戻す処理を追加。
- train notebook を exp072 source check、EDA 実行、生成物確認の構成に更新。
- inference notebook は no-op policy check に更新。

## 再現性メモ

- seed policy: `no_rng_used_except_stable_representative_sampling`
- stochastic components: なし。PF/Beam は新規生成しない。
- CPU/GPU runtime: CPU only。
- source artifact: exp072 v2 `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- known exp072 v2 gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- 実行時に raw SHA と decompressed content SHA を summary JSON に記録する。

## 実装後の検証

```bash
uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py experiments/exp083_pf_beam_true_tvt_2d_well_eda/settings.py
uv run python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py experiments/exp083_pf_beam_true_tvt_2d_well_eda/settings.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_experiment.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook inference --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-infer --title "exp083 pfbeam true tvt eda infer" --run-on-push --strict
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/update_experiment_summary.py
```

- `ruff check`: pass
- `py_compile`: pass
- `validate_experiment`: pass
- `prepare_kaggle_notebooks` train: pass
- `prepare_kaggle_notebooks` inference: pass
- train metadata: `enable_gpu=false`, `enable_internet=false`, `kernel_sources=["kentookumura/exp072-exp063-full-replay-feature-cache-train"]`
- `update_experiment_summary`: pass。`experiment_summary.md` に exp083 行と `exp072 -> exp083` lineage が追加された。
- 合成 exp072 型 CSV で `target` delta 復元、candidate materialize、well summary、metrics 出力を確認。ローカル uv 環境には `matplotlib` がないため PNG 作成は Kaggle 実行で確認する。

## 次のアクション

Kaggle 実行:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
kaggle kernels pull kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-pull/exp083-pfbeam-true-tvt-eda-train -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v1
```

- v1 push: success。kernel id `kentookumura/exp083-pfbeam-true-tvt-eda-train`。
- v1 status: `RUNNING` 後、logs で失敗確認。
- v1 failure: exp072 cache の `well` column が mixed dtype で、`well_summary.set_index("well_id").to_dict(orient="index")` が duplicate index error。原因は読み込み時の dtype 未固定。
- fix: `pd.read_csv(..., dtype={"id": str, source.well_column: str}, low_memory=False)` と `frame[source.well_column].astype(str)` を追加。

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v2
```

- v2 push: success。
- v2 status: completed via logs.
- rows / wells: 3,783,989 / 773
- plot count: 70
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary PF mean well RMSE: 10.844610073716387
- primary Beam mean well RMSE: 12.587073725465121
- anchor mean well RMSE: 12.812478851769727
- output synced:
  - `artifacts/pf_beam_true_tvt_2d_well_eda_summary.json`
  - `artifacts/pf_beam_true_tvt_2d_well_eda_well_summary.csv`
  - `artifacts/pf_beam_true_tvt_2d_well_eda_plot_manifest.csv`
  - `artifacts/pf_beam_true_tvt_2d_well_eda_plots.zip`
  - `artifacts/exp083-pfbeam-true-tvt-eda-train.log`

## v3 clean plot

ユーザー確認で full plot が見づらく、変動が大きい線は主に `sc_ens` / `hyb` 側と判断。v3 では plot columns を次の 5 本に限定した。

- true TVT
- PF ANCC
- Beam mean
- Likelihood PF mean
- last anchor

変更:

- `eda.output_prefix`: `pf_beam_true_tvt_2d_well_eda_clean`
- `eda.plot_columns.primary`: `last_anchor_tvt`, `pf_ancc`, `beam_mean`, `likpf_mean`
- `eda.plot_columns.secondary`: `[]`
- plot colors: PF ANCC blue、Beam mean orange、Likelihood PF mean green、last anchor gray dashed

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_experiment.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v3
```

- v3 push: success。
- v3 status: completed.
- rows / wells: 3,783,989 / 773
- clean plot count: 70
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary PF mean well RMSE: 10.844610073716387
- primary Beam mean well RMSE: 12.587073725465121
- anchor mean well RMSE: 12.812478851769727
- output synced at the time: clean representative summary / well summary / manifest / plots zip / v3 log. Local plot files were later replaced by the v11 diagnostic outputs.

## v4 clean all-well plot

ユーザー依頼により、v3 と同じ 5 本の clean plot を全 well に対して作成した。

変更:

- `eda.output_prefix`: `pf_beam_true_tvt_2d_well_eda_clean_all`
- `eda.max_plots`: `null`
- `eda.plot_all_wells`: `true`
- plot reason: 全て `all_wells`

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_experiment.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v4
```

- v4 push: success。Kernel version 4。
- v4 status: completed.
- rows / wells: 3,783,989 / 773
- clean all-well plot count: 773
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary PF mean well RMSE: 10.844610073716387
- primary Beam mean well RMSE: 12.587073725465121
- anchor mean well RMSE: 12.812478851769727
- Kaggle CLI output は page 分割され、展開済み PNG は最初の page までしか直接取得されなかった。`plots.zip` は取得でき、zip 内 773 PNG を確認後、local artifact directory に展開した。
- output synced at the time: clean all-well summary / well summary / manifest / plots zip / expanded plot directory / v4 log. Local plot files were later replaced by the v11 diagnostic outputs.

## 次のアクション

1. `pf_beam_disagreement_error_map` で exp083 well summary / plot manifest を読み、PF-vs-Beam 不一致と exp073 OOF error の関係を bucket 化する。

## v5-v7 ANCC/Z physical decomposition plot

ユーザー依頼により、PF 生成結果ではなく true TVT の急変が `Z` 変化、raw train-only `ANCC`、または `ANCC - Z` の合成に対応するかを読むための physical decomposition view を追加した。

追加 plot 内容:

- 上段: `true TVT`、`PF Z`、`last anchor`
- 中段: raw train horizontal CSV の `Z`、`ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA`
- 下段: `dTVT/dMD`、`-dZ/dMD`、`dANCC/dMD`、`d(ANCC - Z)/dMD`
- `abs(dTVT/dMD)` 上位点を縦線で表示

実装メモ:

- raw formation columns は train-only なので EDA 専用。推論特徴、提出候補、hard rule には直接使わない。
- exp072 cache の `id={well}_{row_idx}` から raw train row index を復元し、plot 対象 well ごとに raw CSV を join する。
- v5 は full frame へ raw context を一括 join して kernel died。v6 で per-plot well join に変更。
- v6 は `ANCC` が全欠損の well で停止したため、v7 では `MD` / `Z` だけを必須にし、formation 欠損 well でも plot を出すよう修正。

実行:

```bash
uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
uv run python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py
uv run python scripts/validate_experiment.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
kaggle kernels status kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels logs kentookumura/exp083-pfbeam-true-tvt-eda-train
```

- v7 push: success。Kernel version 7。
- v7 status: COMPLETE。
- rows / wells: 3,783,989 / 773
- ANCC/Z decomposition plot count: 773
- source gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- source decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary PF mean well RMSE: 10.844610073716387
- primary Beam mean well RMSE: 12.587073725465121
- anchor mean well RMSE: 12.812478851769727
- physical context: raw train columns joined per plot well from `/kaggle/input/competitions/rogii-wellbore-geology-prediction/train`
- output download: user request により途中停止。停止前に manifest と zip は取得済み。
- local zip integrity: `unzip -t` PASS、zip 内 PNG count 773。
- output synced at the time: ANCC/Z decomposition manifest and plots zip. Local plot files were later replaced by the v11 diagnostic outputs.

## v8-v11 current diagnostic plot

ユーザー確認により、v7 の3段 decomposition view ではなく、v4 clean all-well plot を基準に診断情報を重ねる形へ変更した。

変更:

- v8: clean all-well plot に `PF Z` と raw physical background を追加。
- v9: `ANCC` だけでなく `ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` を formation band として薄く塗り分け。
- v10: `Z` 背景を追加。
- v11: 下段に `dZ/dMD` panel を追加。

現在のローカル参照成果物:

- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plot_manifest.csv`
- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots.zip`
- `artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots/`

ローカル同期:

```bash
find experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts -maxdepth 1 \( -name '*plot_manifest.csv' -o -name '*plots.zip' -o -name '*plots' -o -name '*summary.json' -o -name '*well_summary.csv' \) -exec rm -rf {} +
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v11
unzip -t /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v11/artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots.zip
unzip -q -o experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots.zip -d experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_plots
```

- v11 manifest rows: 773。
- v11 zip integrity: PASS。
- v11 zip PNG count: 773。
- v11 extracted PNG count: 773。
- Kaggle CLI は manifest と zip 取得後、個別PNGの取得が遅かったため停止した。ローカルの個別PNGは検証済みzipから展開した。

## v12 prediction start line

ユーザー確認により、test の `TVT_input` 領域は使わず、train-side EDA plot に prediction start の縦線だけを追加する方針にした。

変更:

- `eda.prediction_start_line` を追加。
- 現在の x 軸は exp072 feature cache の `md_since` なので、prediction start は `md_since=0` として赤の点線で表示する。
- 下段 `dZ/dMD` panel が有効な場合も同じ x 位置に縦線を表示する。
- output prefix を `pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_predstart` に変更し、v11 生成物を上書きしない。
- test の `TVT_input` は読み込まず、描画にも使わない。

実行:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels logs kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels pull kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-pull/exp083-pfbeam-true-tvt-eda-train-v12 -m
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p /tmp/kaggle-output/exp083_pf_beam_true_tvt_2d_well_eda/train_v12
```

結果:

- Kaggle kernel version 12 として push 成功。
- kernel id: `kentookumura/exp083-pfbeam-true-tvt-eda-train`
- URL: https://www.kaggle.com/code/kentookumura/exp083-pfbeam-true-tvt-eda-train
- metadata pull は成功し、`enable_gpu=false`、`machine_shape=None`、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train` を確認。
- CLI logs は本文なし。`logs -f` は既知挙動と同様に空で終了。
- 初回の `kaggle kernels output` は反映前でローカル取得ファイル 0 件。
- ユーザーが Kaggle 側の完了を確認。ローカルへの結果 download は不要との指示のため、追加 output 取得は実施しない。

## v13 prediction start line fallback

ユーザー確認により、v12 plot に prediction start の縦線が見えないことを確認。

原因:

- v12 は `md_since=0` が plot 範囲内にある場合だけ `axvline` を描いていた。
- exp072 feature cache の plot は hidden tail 側だけを描く well があり、最小 `md_since` が 0 より大きい場合は `md_since=0` が x 軸範囲外になる。

変更:

- `eda.prediction_start_line.fallback_to_first_x: true` を追加。
- `md_since=0` が左端より前にある場合は、各 well の最初の描画 x 位置に prediction start 線を表示する。
- output prefix を `pf_beam_true_tvt_2d_well_eda_clean_all_pfz_formation_bands_z_dzdmd_predstart_v13` に変更する。
- test の `TVT_input` は引き続き読み込まず、描画にも使わない。

## v14 known-prefix replay overlay

ユーザー確認により、exp083 の既存 plot に対して known `TVT_input` 区間で生成した PF/Beam を重ねる形へ修正した。

補足:

- exp169 本体は known prefix holdout replay を内部で実行していたが、ローカル取得済み output は `prefix_offsets.csv` の offset summary で、row-level trajectory は保存されていなかった。
- そのため v14 では exp083 側で、exp169 と同じ known prefix 末尾 256 rows holdout 条件の PF/Beam replay を selected wells だけ再実行して overlay した。
- 全 known TVT 区間の replay ではなく、exp169 と同じ `prefix_holdout_rows=256` の known prefix 末尾区間である。

変更:

- `public_notebook_replay_audit.py` を exp083 に追加し、known prefix holdout replay に使用。
- `eda.selected_wells` を追加し、対象を `91b301ce`, `ba48188d`, `fef8af96`, `1b1eba53`, `86454a6f`, `4e050c92` に限定。
- `eda.known_replay_overlay` を追加。
- exp083 plot に prediction start 前の known prefix 行を追加し、`known_replay_pf_ancc`, `known_replay_pf_z`, `known_replay_beam_mean`, `known_replay_likpf_mean` を破線 overlay。
- output prefix を `pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14` に変更。

実行:

```bash
python3 -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py experiments/exp083_pf_beam_true_tvt_2d_well_eda/public_notebook_replay_audit.py
uv run ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/pf_beam_true_tvt_eda.py experiments/exp083_pf_beam_true_tvt_2d_well_eda/public_notebook_replay_audit.py --select F821,F722,F823
uv run python scripts/validate_experiment.py --experiment exp083_pf_beam_true_tvt_2d_well_eda
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp083_pf_beam_true_tvt_2d_well_eda --notebook train --kernel-id kentookumura/exp083-pfbeam-true-tvt-eda-train --title "exp083 pfbeam true tvt eda train" --run-on-push --strict
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/train
kaggle kernels status kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels logs kentookumura/exp083-pfbeam-true-tvt-eda-train
kaggle kernels output kentookumura/exp083-pfbeam-true-tvt-eda-train -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/train_v14_known_prefix_overlay
```

結果:

- Kaggle kernel version 13 として push 成功。
- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/train_v14_known_prefix_overlay`
- plot count: 6
- known replay overlay: 6 wells ok、1,536 rows、各 well 256 replay rows
- PNG: `artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plots/*.png`
- manifest: `artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_plot_manifest.csv`
- summary: `artifacts/pf_beam_true_tvt_2d_well_eda_known_prefix_replay_overlay_v14_summary.json`

確認:

- `file` で全 6 PNG が `1750 x 1162` と確認できた。
- manifest は selected 6 wells の各行に `known_replay_overlay` を持ち、`overlay_rows=256`。
- `selected_wells__fef8af96.png` と `selected_wells__91b301ce.png` を目視し、exp083 v11 系の plot に known replay PF/Beam 破線が prediction start 前の known prefix 区間へ重なっていることを確認した。

## v12 ML OOF + known raw TVT probe 別 notebook

ユーザー依頼により、既存正規 train notebook は上書きせず、v12 prediction-start plot を拡張する別 notebook を追加した。対象は全 773 well。`TVT_input` の既知 prefix 区間はプロット対象外とし、exp072 feature cache rows だけを描画する。

追加ファイル:

- `exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py`
- `exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb`
- steering: `.steering/20260705-exp083-v12-ml-oof-known-tvt-probe/`

内容:

- exp072 full replay feature cache を読み、`true_tvt = last_known_tvt + target` と PF/Beam candidates を復元する。
- exp148 train output の `exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz` から `variant=learned_likelihood_confidence_addonly`、`mode=gpu_repro_guard_dp_threads8`、`model=lgb_mean` の OOF を抽出し、`exp148 ML OOF lgb_mean` として overlay する。
- raw train horizontal CSV を `id={well}_{row_idx}` の row index で join し、同じ feature-cache rows 上の `TVT` を `known raw TVT probe` として scatter overlay する。
- v12 と同じ `md_since=0` prediction start line を表示し、x=0 が plot 範囲外の場合は左端に fallback 表示する。
- `-Z scaled`、formation band、下段 `dZ/dMD` は診断背景として維持する。

Kaggle 実行時の input source:

- `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- `kentookumura/exp148-train`
- competition raw train data

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_v12_ml_oof_known_tvt_probe_jsoncheck.out
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass

## v12 ML OOF + exp209 HMM +/-2sigma band 修正

ユーザー指示により、HMM の 2sigma range を薄い band として TVT panel に追加した。直前の v24 no-formation 版は Kaggle status/logs で COMPLETE を確認済み。output は取得していない。

修正:

- exp209 HMM mean の周りに `hmm_mean_tvt +/- 2*hmm_std` を薄い紫色の `fill_between` band として描画する。
- band は `hmm_std` が finite かつ 0 以上の点だけを使い、同じ `md_since` は median 集約する。
- HMM mean 線は従来通り残し、`likPF/HMM blend` は引き続き描画しない。
- HMM +/-2sigma band の min/max も TVT panel の y-axis range に含める。
- manifest に `exp209_hmm_2sigma_segments`, `exp209_hmm_2sigma_points`, `exp209_hmm_2sigma_min`, `exp209_hmm_2sigma_max` を追加した。
- summary に `hmm_std_column`, band style, formula、calibration 注意書きを追加した。
- 地層 line / band は引き続き描画しない。
- PNG 保存名は引き続き `{well}.png` で、`all_wells__` prefix は使わない。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all` に変更した。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
LC_ALL=C rg -n "[^[:ascii:]]" experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass
- `ruff --select F821`: pass
- source `.py` ASCII check: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- Package notebook grep: `hmm_2sigma` / `+/-2sigma` / `2sigma_noformation` あり、`all_wells__` / `blend_likpf` / `formation_axis_context` はなし。
- Kaggle に v25 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。
- 実行中 logs は空。
- ユーザー指示により監視だけ停止した。Kaggle 側実行は継続中。

## v12 ML OOF + exp209 HMM raw-formation background 修正

ユーザー確認により、背景に薄く表示している地層の scale がずれて見えること、`likPF/HMM blend` は不要であることを確認した。

修正:

- 地層背景 `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` を plot y-range への共通 min-max rescale 対象から外し、raw TVT/depth scale のまま描画するようにした。
- 地層 label から `scaled` 表記を削除した。
- exp209 overlay は `hmm_mean_tvt` のみを描画するようにし、`blend_likpf_hmm_w500` は読み込み、描画、title、manifest、summary から削除した。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_rawformation_all` に変更した。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- Kaggle に v22 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。
- 実行中 logs は空。完了判定は後続の status/logs で行う。

v22 完了確認:

- ユーザーから完了連絡あり。
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` で logs を確認した。
- v22 は 773 well の plot 生成を完了。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_rawformation_all`
- manifest rows: 773。
- exp209 HMM rows per well min/max: 407 / 10052。
- exp209 HMM mean points per well min/max: 407 / 10052。
- exp209 HMM mean TVT range: 10047.468 / 12888.823。
- output は取得していない。

## v12 ML OOF + exp209 HMM formation Z-to-TVT 修正

ユーザー確認により、v22 の地層背景が全く表示されていないことを確認した。原因は formation 列 `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` が `TVT` ではなく `Z` と同じ座標系であり、raw 値のまま TVT 軸に描くと -9000 付近になって画面外へ出ること。

修正:

- formation 列を `formation_tvt = raw_TVT + raw_Z - formation_Z` で TVT 軸へ変換するようにした。
- common min-max rescale は使わない。
- TVT y-axis に、通常の予測範囲を挟む近傍 formation bracket だけを追加するようにした。
- manifest に `formation_axis_context_points`, `formation_axis_context_min`, `formation_axis_context_max` を追加した。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_formationztvt_all` に変更した。
- `likPF/HMM blend` は引き続き読み込み・描画・title・manifest・summary から削除済み。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- Kaggle に v23 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。
- 実行中 logs は空。完了判定は後続の status/logs で行う。

v23 完了確認:

- ユーザーから完了連絡あり。
- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`。
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` で logs を確認した。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_formationztvt_all`
- manifest rows: 773。
- exp209 HMM rows per well min/max: 407 / 10052。
- exp209 HMM mean points per well min/max: 407 / 10052。
- Formation TVT context points per well min/max: 2 / 4。
- PNG はまだ `all_wells__{well}.png` 名で生成されていた。
- output は取得していない。

## v12 ML OOF + exp209 HMM no-formation plot 修正

ユーザー確認により、地層 plot は不要、PNG 名の `all_wells` も不要であることを確認した。

修正:

- 地層境界 line と地層 filled band を完全に描画対象から外した。
- formation context による y-axis range 調整と manifest 列を削除した。
- `-Z likPF minmax` guide は地層ではないため引き続き残した。
- PNG 保存名を `all_wells__{well}.png` から `{well}.png` に変更した。zip 内の名前も `{well}.png` になる。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_noformation_all` に変更した。
- `likPF/HMM blend` は引き続き読み込み・描画・title・manifest・summary から削除済み。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- Package notebook grep: `all_wells__` / formation context / `likPF/HMM blend` 関連文字列なし。
- Kaggle に v24 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。
- 実行中 logs は空。完了判定は後続の status/logs で行う。

## v12 plot exp215 full-tail MTP overlay 差し替え

ユーザー指示により、v12 plot notebook の path overlay を exp212 から exp215 output に差し替えた。

変更:

- 使用する追加 input source を `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train` に変更した。
- primary artifact を `exp215_mtp_full_tail_heatmap_path_generator_probe_full_grid_candidate_paths.csv.gz` に変更した。
- exp215 full-grid paths の `path_rank` 1-5 を TVT panel に描画する。rank1 は濃い紫線、rank2-5 は薄い紫線。
- exp215 `weighted_tvt_pred` を rank1 rows から x ごとに median 集約し、オレンジ破線 `exp215 learned weighted path` として描画する。
- x 軸は exp215 artifact の `md_since`、y 軸は `tvt_pred` / `weighted_tvt_pred`。
- rank1 path と weighted path は TVT panel の表示 y 範囲に含め、スケールずれを隠さない。
- title / manifest / summary は exp215 `existing_plus_learned_mtp_topk` と `existing_union` の candidate-union readout を使う。
- exp215 の `summary.json` を source SHA と parent summary として記録する。exp212 にあった source window coverage CSV / contract metrics CSV は exp215 にはないため読まない。

exp215 親実験ログ上の代表値:

- full-grid rows: 18,919,945。
- unique row ids: 3,783,989。
- wells: 773。
- path ranks: 5。
- row coverage vs cache: 1.0。
- fallback unique row rate: 0.0。
- existing union oracle RMSE: 7.434029932。
- learned MTP top5 only oracle RMSE: 32.333142886。
- learned MTP weighted oracle RMSE: 59.272141581。
- existing + learned MTP top5 oracle RMSE: 5.113654814。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass。
- `ruff --select F821`: pass。
- `jupytext --to ipynb`: pass。
- `jupytext --to ipynb --test`: pass。
- notebook JSON check: pass。
- package notebook JSON check: pass。

未実施:

- Kaggle push / run はこれから実行する。

Kaggle 実行完了確認:

```bash
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- version: 21。
- status: `KernelWorkerStatus.COMPLETE`。
- output archive: 取得していない。logs / status のみ確認。
- input source:
  - `/kaggle/input/exp209-joint-exact-parity-train/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
  - `/kaggle/input/notebooks/kentookumura/exp148-train/artifacts/exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz`
  - `/kaggle/input/exp209-joint-exact-parity-train/artifacts/exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz`
  - `/kaggle/input/exp209-joint-exact-parity-train/artifacts/exp209_vs_exp072_exp205_by_well_delta.csv`
  - `/kaggle/input/exp209-joint-exact-parity-train/artifacts/exp209_vs_exp072_exp205_overall_metrics.csv`
  - `/kaggle/input/exp209-joint-exact-parity-train/artifacts/exp209_vs_exp072_exp205_summary.json`
- rows:
  - PF/Beam: 3,783,989。
  - exp148 `lgb_mean` OOF: 3,783,989。
  - exp209 enriched HMM: 3,783,989。
  - joined: 3,783,989。
- wells:
  - source: 773。
  - plotted: 773。
- output:
  - manifest: `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_all_plot_manifest.csv`
  - plots dir: `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_all_plots`
  - plots zip: `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_all_plots.zip`
  - summary: `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_all_summary.json`
- manifest rows: 773。
- `TVT_input` prefix plotted: false。
- prediction-start line plotted: false。
- known TVT probe plotted: false。
- `-Z` likPF minmax status: `{'ok': 773}`。
- `-Z` likPF minmax coverage min: 1.0。
- exp209 HMM rows per well min/max: 407 / 10052。
- exp209 HMM mean points per well min/max: 407 / 10052。
- exp209 best candidate: `blend_likpf_hmm_w500`。
- exp209 global metrics:
  - `exp072_likpf_mean` RMSE: 11.594897672217703。
  - `hmm_mean_tvt` RMSE: 11.938287096143716。
  - `blend_likpf_hmm_w500` RMSE: 10.269696146642758。
- exp209 best blend RMSE range by well: 0.4892217739793377 / 45.87981738453106。
- exp209 parent by-well delta summary:
  - improved wells: 451。
  - worsened wells: 322。
  - max regression well: `b19b0395`。
  - max regression RMSE: 48.31578609448495。
- source SHA in summary:
  - exp209 enriched HMM gzip `69b9038394dc5a0e9da5db9721f74c35aed60d676caf0d4b56a3cb01825a62c6`
  - exp209 enriched HMM decompressed `ee3b548b0d38f78966742542e86fa31b7e64698d4762b924c924a5895d2ee3f4`
  - exp209 by-well delta `a600542874652af8a9c60c94872f751a0c140cf384835dd1eccd5f9eb4e787ff`
  - exp209 overall metrics `2595072d11363f7b4fd60a082aa98f50dd26b45a36bba4651c5f9287feef0956`
- notes:
  - exp209 HMM mean and best likPF/HMM blend are plotted from enriched HMM output.
  - exp209 x uses `md_since` joined from exp072 plot frame by `id/well`.
  - exp209 title metrics are train-side by-well direct-comparison readouts, not hidden-test prediction scores.
  - No model training, PF/Beam regeneration, inference, or submission is performed.

## v12 plot exp209 HMM overlay 差し替え

ユーザー指示により、exp215 output は描画対象から外し、v12 plot notebook の path / HMM overlay を exp209 HMM output に差し替えた。

変更:

- 使用する追加 input source を `kentookumura/exp209-joint-exact-parity-train` に変更した。
- exp215 input source `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train` は Kaggle package metadata から削除した。
- primary artifact を `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz` に変更した。
- `hmm_mean_tvt` を紫線 `exp209 HMM mean` として描画する。
- exp209 best candidate `blend_likpf_hmm_w500` をオレンジ破線 `exp209 likPF/HMM blend50` として描画する。
- x 軸は exp209 enriched cache の `id/well` を exp072 plot frame に join した `md_since` を使う。
- title / manifest / summary は `exp209_vs_exp072_exp205_by_well_delta.csv` から well ごとの HMM RMSE、best blend RMSE、likPF RMSE を表示する。
- summary には `exp209_vs_exp072_exp205_overall_metrics.csv` と `exp209_vs_exp072_exp205_summary.json` を記録する。
- exp209 は row-level HMM mean / blend feature output で、full path candidate output ではないことを summary に残す。

exp209 親実験ログ上の代表値:

- rows / wells: 3,783,989 / 773。
- best candidate: `blend_likpf_hmm_w500`。
- best RMSE: 10.269696146642758。
- exp072 `likpf_mean` RMSE: 11.594897672217703。
- HMM mean RMSE: 11.938287096143716。
- best delta vs exp072 `likpf_mean`: -1.3252015255749452。
- HMM feature parity: PASS。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass。
- `ruff --select F821`: pass。
- `jupytext --to ipynb`: pass。
- `jupytext --to ipynb --test`: pass。
- notebook JSON check: pass。
- package notebook JSON check: pass。

未実施:

- Kaggle push / run はこれから実行する。

## v12 ML OOF + exp212 full-grid path overlay

ユーザー指示により、exp210 full-well path overlay を exp212 `heatmap_mdn_full_grid_path_generation_probe` の output に差し替えた。

変更:

- primary artifact を `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz` に変更した。
- candidate union / source coverage / contract metrics も exp212 の `localtopk10` output に変更した。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp212_fullgridpath_all` に変更した。
- Kaggle package metadata の kernel source を `kentookumura/exp212-hmdn-full-grid-path-generation-train` に差し替えた。
- exp212 path の x 軸は `md_since`、y 軸は `tvt_pred` とした。`md_from_ps` は source coverage 記録用に保持する。
- exp210 版で残っていた TVT trusted range による path 点の y-filter を廃止した。
- exp212 rank1 の min/max を TVT panel の display y range に含め、full-grid line の scale mismatch が見えるようにした。
- manifest に `exp212_full_grid_rank1_raw_points`, `exp212_full_grid_rank1_source_points`, `exp212_full_grid_rank1_fallback_points`, `exp212_full_grid_rank1_fallback_rate`, `exp212_full_grid_rank1_coverage_rate`, `display_y_min`, `display_y_max` を追加した。
- summary notes に、exp212 は direct prediction score ではなく train-side candidate-union oracle diagnostic であることを明記した。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local notebook JSON check: pass
- package notebook JSON check: pass
- exp210/full-well slug 参照が source/package metadata から消えていることを `rg` で確認した。
- Kaggle push / run はこれから実行する。

## v12 ML OOF + exp210 full-well path overlay 差し替え

ユーザー指示により、未実行だった exp207 overlay 版から exp210 `heatmap_mdn_full_well_path_generation_probe` の full-well candidate path output に差し替えた。

修正:

- exp207 `stitched_path_rows.csv.gz` 読み込みを削除し、exp210 primary artifact `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz` を読むように変更。
- 描画列は `md_from_ps` を x、`tvt_pred` を y、`path_rank` を rank とする。exp210 artifact は full-well path contract で x 座標を持つため、exp072 feature-cache rows への `id/well` join は行わない。
- `path_rank` 1-5 を描画し、rank1 を濃い紫線 + marker、rank2-5 を薄い紫線にした。
- by-well title / manifest / summary は exp210 `candidate_union_by_well.csv` の `existing_oracle_rmse`, `union_oracle_rmse`, `new_best_candidate_rate` を表示する。
- summary には exp210 full-well candidate path / candidate union / source window coverage / contract metrics の path と SHA を記録する。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp210_fullwellpath_all` に変更。
- Kaggle package metadata の kernel source を `kentookumura/exp210-hmdn-full-well-path-generation-train` に差し替えた。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/kernel-metadata.json
rg -n "exp207|EXP207|hmdn-path-stitch" experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/kernel-metadata.json experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- `kernel-metadata.json` JSON check: pass
- exp207 / old path-stitch source 参照なし。

Kaggle push / run はこの後に実施する。

Kaggle v18 実行:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-exp210-v18 -m
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kaggle kernel version 18。
- metadata pull: success。`kernel_sources` は `exp072-exp063-full-replay-feature-cache-train`、`exp148-train`、`exp210-hmdn-full-well-path-generation-train`。
- ユーザー指示により logs 監視だけ停止。Kaggle 実行自体は継続。

完了後確認:

```bash
kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe
kaggle kernels output kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/exp083_v18_manifest --file-pattern '.*(manifest|summary).*'
```

- status: completed by logs。
- plot scope: 773 wells。
- exp210 artifact source: `/kaggle/input/exp210-hmdn-full-well-path-generation-train/artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz`
- exp210 rows: 8,137,310。
- exp210 path wells: 773。
- exp210 path ranks: 1-5。
- exp210 contract: `unique_row_ids=1,627,462`、`row_coverage_rate_vs_cache=0.4300916308160515`、`path_count=3865`。
- plots generated: 773 PNG + zip。
- summary SHA for exp210 full-well candidate path decompressed: `f22808f0c0af8cc8a2953680284db9d8564fcecfa401a8921c6130e29f8509f0`。

ユーザー確認: `full pathになっているようには見えない`。

切り分け:

- manifest では `exp210_full_well_rows` はほぼ 10,555 rows/well、つまり 2,111 rows × 5 ranks。
- 一方、exp083 plot の `source_rows` は well により 3,000-10,000 rows 程度あり、exp210 の `md_from_ps` は多くの well で 1..2111 までしかない。
- さらに plot notebook は TVT panel の scale を壊さないため、true/PF/Beam/ML/-Z guide から作った trusted y range 外の exp210 path points を描画前に除外している。
- manifest 分布:
  - `exp210_full_well_rank1_points` median 1,725、min 0、max 2,111。
  - 77 wells は rank1 plotted points 0。
  - 133 wells は rank1 plotted points < 100。
  - rank1 plotted points / exp083 source rows median 0.322。
- 例:
  - `00e12e8b`: exp210 rank1 raw rows 2,111、x range 1..2111、raw TVT 11359.70..11754.98。plot trusted y range 11579.05..11632.27 に入る点は 33 のみ。
  - `febb4411`: exp210 rank1 raw rows 2,111、raw TVT 12645.35..12710.11。plot trusted y range 12575.39..12624.92 に入る点は 0。

解釈:

- exp210 の `full-well` は README/config の記述どおり、後続 selector 用に exp099 candidate-cache と一致する covered rows へ限定した full-well candidate path contract であり、exp083 plot の全 `md_since` 区間を覆う full trajectory ではない。
- 現在の plot はこの exp210 contract を正しく読んでいるが、`full path` として期待される見た目にはならない。
- 真に exp083 plot 全区間を覆う path を見たい場合は、exp210 artifact の可視化では不十分。exp208/exp202 の dense local windows から、exp072 feature-cache row grid 全体へ再 stitch / decode する別 generator が必要。

## v12 exp207 stitched path output 差し替え

ユーザー指示により、v12 plot notebook の heatmap-MDN overlay を exp202 rank1 center path から exp207 `heatmap_mdn_overlapping_window_path_stitch_probe` の出力へ差し替えた。Kaggle push / run はまだ実行していない。

変更:

- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp207_stitchedpath_all` に変更した。
- 入力を exp207 artifacts に変更した。
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_stitched_path_rows.csv.gz`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_by_well.csv`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_metrics.csv`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_source_window_coverage.csv`
- `stitched_path_rows` を `id/well` で exp072 feature-cache rows に join し、`md_since` 上に rank1-3 stitched path を描く。
- rank1 は濃い紫線、rank2-3 は薄い紫線とした。
- TVT panel y-axis は引き続き true/PF/Beam/ML/-Z guide から固定し、exp207 out-of-range point が autoscale を壊さないようにした。
- title / manifest / summary は exp207 covered-row union oracle RMSE、existing oracle RMSE、new-best rate を出す。
- Kaggle package metadata の kernel source を `kentookumura/exp202-heatmap-mdn-candgen-train` から `kentookumura/exp207-hmdn-path-stitch-train` に変更した。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_notebook_check.json
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_pkg_notebook_check.json
python3 -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/kernel-metadata.json >/tmp/exp083_kernel_metadata_check.json
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local `.ipynb` JSON check: pass
- package `.ipynb` JSON check: pass
- `kernel-metadata.json` JSON check: pass

## v12 exp202 saved path segment overlay 修正

ユーザーが exp202 output に window 内 path を保存するよう修正したため、plot notebook 側も exp202 の保存済み path artifact を読む形に変更した。

修正:

- exp202 source を `kentookumura/exp202-heatmap-mdn-candgen-train` とし、以下を解決する。
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`
- 既存の `pred_top1_tvt` center-line overlay ではなく、NPZ の `pred_tvt_path` を plot する。
- rank1 path segment は紫の濃線、rank2-10 は薄い紫線として描画する。
- `horizontal_offsets == 0` を center として、`md_path - center_md + md_since_prefix` で各 local path segment の plot x 座標を復元する。
- plot frame は従来通り feature-cache の plot 対象行に限定し、TVT_input known prefix rows は追加しない。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp202_pathsegments_all` に変更した。
- summary / manifest に exp202 path sample 数、rank1/rank2-10 segment 数、path artifact SHA を記録する。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; p='experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb'; json.load(open(p)); print('json_ok', p)"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- Kaggle package `.ipynb` JSON check: pass

Kaggle v17:

- `kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe`
- version: 17
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- push 後 status: `KernelWorkerStatus.RUNNING`
- metadata pull: `/tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v17`
- 完了後 status: `KernelWorkerStatus.COMPLETE`
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp202_rank1centerpath_all`
- manifest rows: `773`
- exp202 rank1 center path points per well min/max: `0` / `14`
- exp202 rank1 center path TVT range in manifest: `10041.417` - `12862.111`
- 代表画像 `all_wells__00e12e8b.png` を 1 枚だけ取得して確認した。
- `00e12e8b` は exp202 rank1 center candidates が trusted y range 外のため 0 点になり、v16 で見えていた紫線による y-axis scale 破壊は解消した。

Kaggle v16:

- `kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe`
- version: 16
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- push 後 status: `KernelWorkerStatus.RUNNING`
- metadata pull: `/tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v16`
- 10 分程度の follow logs と通常 logs はログ本文なし。status は `RUNNING` のまま。
- ユーザー指示により logs 監視だけ停止した。Kaggle 実行自体は継続。

## v16 exp202 stitched guide scale 異常の見直し

ユーザー確認により、v16 完了後も「一つだけスケールがおかしい」と判明した。status/logs を確認し、v16 は正常完了していた。

確認:

- status: `KernelWorkerStatus.COMPLETE`
- v16 logs:
  - output prefix: `...exp202_stitchedpath_all`
  - manifest rows: `773`
  - exp202 stitched rank1 points per well min/max: `407` / `1728`
  - `-Z likPF minmax` は全 well `ok`
- 代表画像 `all_wells__00e12e8b.png` を 1 枚だけ取得して確認した。
- 異常なのは紫の `exp202 stitched path rank1` で、true TVT が約 11600 ft 周辺なのに、紫線だけ 11341-11774 ft まで広がって y-axis scale を壊していた。
- local exp202 artifact から再計算しても、`00e12e8b` の rank1 stitched full path は `11341.41` - `11773.83` ft、true center TVT は `11587.65` - `11616.30` ft だった。
- exp202 実装上、`pred_tvt_path` は absolute TVT だが、full local path logits は center 以外の step が大きく飛ぶことがある。center prediction は `center_pred_tvt` として scale が合っている。

修正方針:

- full `pred_tvt_path` 全 step の stitching は廃止する。
- `candidate_path_rank_index.csv.gz` の `rank == 1` / `center_pred_tvt` を、`candidate_path_samples.csv.gz` の `md_since_prefix` で sample center 順につなぐ。
- plot label は `exp202 rank1 center path` とし、full-well trajectory ではなく sparse center candidate path であることを summary に明記する。
- rank1 center prediction 自体も外れる well があるため、TVT panel の y-axis は true TVT、PF/Beam、exp148 ML OOF、`-Z likPF minmax` の trusted series から固定する。
- exp202 center candidates は trusted y range 内だけ描画し、out-of-range candidate が panel autoscale を壊さないようにした。
- manifest に `exp202_path_rank1_center_points`, `exp202_path_rank1_center_min`, `exp202_path_rank1_center_max` を追加し、次回 logs で y range を確認できるようにした。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; p='experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb'; json.load(open(p)); print('json_ok', p)"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- Kaggle package `.ipynb` JSON check: pass

Kaggle v15:

- `kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe`
- version: 15
- push 後 status: `KernelWorkerStatus.RUNNING`
- logs で実行時 error を確認。
- error: `AttributeError: 'Pandas' object has no attribute 'prefix_end'`
- 原因: plot notebook の `read_exp202_candidate_path_samples()` が `prefix_end` を `keep_cols` に含めておらず、`itertuples()` の row に `prefix_end` が存在しなかった。

v16 修正:

- `read_exp202_candidate_path_samples()` の `keep_cols` と integer 変換対象に `prefix_end` を追加した。
- 描画時には `prefix_end` が存在しない場合の fallback として `row_center - md_since_prefix` から復元する。

v16 修正後の検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; p='experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb'; json.load(open(p)); print('json_ok', p)"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- Kaggle package `.ipynb` JSON check: pass

Kaggle 実行:

- `kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe`
- version: 14
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- push 後 status: `KernelWorkerStatus.RUNNING`
- metadata pull: `/tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v14`
- 初期 5 分の `kaggle kernels logs -f` と通常 `kaggle kernels logs` はログ本文なし。

## v14 exp202 path segment overlay の見直し

ユーザー確認により、v14 の plot は期待する path 表示になっていないことが判明した。完了後 logs を確認したところ、notebook は正常完了し、exp202 path artifact の読み込みも成功していた。

確認:

- v14 logs で exp202 path npz samples: `10822`
- exp202 path npz topK: `10`
- exp202 path npz horizon: `128`
- exp202 path sample rows per well: `14`
- v14 は各 sample の local 128-row window を rank1 濃線、rank2-10 薄線としてそのまま重ねていた。
- exp202 側の artifact は native full-well trajectory ではなく、validation sample ごとの local window path なので、短い segment 群として見えるのは実装通りだった。

修正:

- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp202_stitchedpath_all` に変更した。
- rank2-10 segment と center marker を描画から外した。
- `horizontal_row_index - prefix_end` で各 local path point を plot の `md_since` 座標へ戻す。
- rank1 (`EXP202_STITCH_RANK=1`) の `pred_tvt_path` だけを使い、同じ integer x に複数 window から予測が来た場合は median 集約する。
- 集約後の `(x, median_tvt)` を x 昇順で1本の紫線として描く。
- summary には、これは visualization aggregation であり、exp202 が native に出力した full-well trajectory でも direct evaluation score でもないことを明記した。

local artifact smoke:

- `000d7d20` / `00bbac68` / `01869cd4` で rank1 stitched points は各 `1728`。
- x range は各 `1..2111`。これは exp202 validation sampling が prefix 後最大 `2048` row の 14 sample center を使い、各 center の local 128-row window を保存しているため。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; p='experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb'; json.load(open(p)); print('json_ok', p)"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- Kaggle package `.ipynb` JSON check: pass

## v12 exp202 path plot feasibility check

ユーザー指示により v12 Kaggle output archive は取得せず、`kaggle kernels status/logs` のみ確認した。v12 は COMPLETE。logs 上では 773 wells を全て plot し、exp202 overlay は `top1 plus top2-10 sparse points` として出力された。

exp202 の保存済み artifact を確認したところ、`validation_predictions.csv.gz` と `heatmap_candidates.csv.gz` には sample center の `pred_top1_tvt` ... `pred_top10_tvt` と `path_step_abs_mean_ft` / `path_step_abs_max_ft` が保存されている。一方、train 実装内で計算している window 内 `pred_path_tvt` 配列そのものは CSV に保存していない。したがって、既存 exp202 output だけで描けるのは「sample center の top1/topK TVT 候補を MD 順に結ぶ path」であり、厳密な window 内 full path を描くには exp202 側で path 配列を保存し直す必要がある。

## v12 exp202 top1 center path line

ユーザー指示により、exp202 overlay の `pred_top1_tvt` を点ではなく線として描画するように修正した。

修正:

- `pred_top1_tvt` を well ごとに `md_since` 昇順で `ax.plot` 接続し、label を `exp202 heatmap top1 path` に変更。
- `pred_top2_tvt` から `pred_top10_tvt` は従来通り薄い sparse point として残した。
- `OUTPUT_PREFIX` を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp202_top1path_all` に変更し、前回 output と混ざらないようにした。
- summary notes に、これは strict full path ではなく保存済み sample center top1 を接続した top1 center path であることを反映した。
- exp202 の新規学習、PF/Beam 再生成、hidden inference、submission は行っていない。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_v12_top1path_jsoncheck.out
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_v12_top1path_pkg_jsoncheck.out
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- source `.ipynb` JSON check: pass
- Kaggle package `.ipynb` JSON check: pass

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v13 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- Kernel version 13 successfully pushed.
- Metadata pull: success.
- status after push: RUNNING.
- ユーザー指示により `logs -f` 監視だけ停止した。Kaggle 実行自体は継続。
- output archive は取得していない。

## v13 exp202 top1 path column audit

ユーザー確認により、v13 完了後に output archive は取得せず `kaggle kernels status/logs` のみ確認した。v13 は COMPLETE。logs 上では output prefix `...simple_exp202_top1path_all` で 773 wells 全てを書き出している。

使用列の確認:

- v13 plot は `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz` の `pred_top1_tvt` を使用している。
- `pred_top1_tvt` は exp202 train 実装内で `combined_score = mode_prob * center_score` により選ばれた mode の center TVT 候補で、window 内 full path ではない。
- exp202 の保存済み CSV には `pred_path_tvt` 配列は保存されておらず、保存されている path 情報は `path_step_abs_mean_ft` / `path_step_abs_max_ft` の summary のみ。
- local exp202 artifact で確認した train-side metrics:
  - rows 10,822、wells 773、各 well 14 sample centers。
  - `pred_top1_tvt` RMSE 約 60.49、MAE 約 39.28、within10 約 0.230。
  - top10 oracle RMSE 約 13.35、MAE 約 7.21、within10 約 0.809。
- したがって、v13 は指定通り `pred_top1_tvt` を線で結んでいるが、この列自体は期待される smooth / correct path ではない可能性が高い。

## v12 ML OOF + -Z typewell-tail minmax guide 修正

Timestamp: 2026-07-05 07:15:59 UTC

ユーザー指示により、v7 の `known anchor + typewell tail hybrid` を廃止し、`-Z` guide を以下の方式へ差し替えた。

- known tail の `TVT_input ~ (-Z)` 一次傾きで、TVT と `-Z` の向きだけを判定する。
- hidden plot rows の `-Z` min/max を source range とする。
- typewell tail の `TVT` min/max を target range とする。
- 向きが正なら source min/max を target min/max に、向きが負なら source min/max を target max/min に対応させる。
- 最後に typewell tail `TVT` min/max へ clip する。

実装メモ:

- plot label: `-Z typewell-tail minmax`
- plot column: `z_typewell_tail_minmax_tvt`
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zminmax_typewelltail_all`
- `TYPEWELL_TAIL_START_QUANTILE=0.60`
- `MIN_TYPEWELL_TAIL_ROWS=8`
- `Z_TVT_DIRECTION_TAIL_POINTS=50`
- `MIN_Z_TVT_DIRECTION_POINTS=8`
- `CLIP_Z_TVT_TO_TYPEWELL_TAIL_RANGE=1`

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v8 -m
rg -n "zminmax|typewell-tail minmax|direction_typewell_tail_minmax|z_typewell_tail_minmax|z_tvt_minmax|z_anchor_typewell|zhybrid|anchor\\+typewell" /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v8
```

- Kaggle version 8 push: success.
- Pull 後の source に minmax 実装が反映されていることを確認。
- old `z_anchor_typewell` / `zhybrid` / `anchor+typewell` は反映 source には残っていない。
- Kaggle output archive は取得していない。
- 2026-07-05 07:23:27 UTC: ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。停止前の `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。
- 2026-07-05 07:39:52 UTC: ユーザー完了連絡後に status/logs のみ確認。output archive は取得していない。
  - `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`
  - output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zminmax_typewelltail_all`
  - plotted wells: 773/773
  - `Z-to-TVT minmax status counts: {'ok': 773}`
  - `Z-to-TVT minmax coverage min: 1.0`
  - `Z-to-TVT known rows min: 851`
  - `Z-to-TVT typewell tail rows min: 15`
  - `Z-to-TVT typewell tail source counts: {'typewell_after_last_known_tvt': 773}`
  - `Z-to-TVT direction counts: {'positive': 476, 'negative': 297}`
  - `Z-to-TVT typewell tail TVT min range: 10055.35 12873.34`
  - `Z-to-TVT typewell tail TVT max range: 10236.85 12991.53`
  - ユーザー目視では `-Z` guide のスケールは依然として PF/Beam / TVT / ML と合わない。`-Z` 単体の min-max scaling は、TVT 絶対座標の補助線としては信頼しない方針にする。

## v12 ML OOF + typewell anchor / -Z likPF minmax guide Kaggle v9 実行開始

記録時刻: 2026-07-05 07:57:58 UTC

ユーザー指示:

- feature candidate としての確認のため、以下 2 本を可視化する。
- 1. last known `TVT_input` を anchor にし、typewell `TVT >= last_known_tvt` tail を hidden 区間 progress 0..1 で補間する guide。
- 2. hidden `-Z` min/max を source range、生成済み `Likelihood PF mean` min/max を target range とする min-max guide。

修正:

- 旧 `-Z typewell-tail minmax` overlay は削除。
- `typewell tail anchored` overlay を追加。
  - `guide = last_known_tvt + (typewell_tail_interp - typewell_tail_start)`
  - hidden progress は `md_since` を 0..1 に正規化。
  - typewell tail は `TVT >= last_known_tvt` を優先し、不足時だけ後半 quantile fallback。
- `-Z likPF minmax` overlay を追加。
  - known tail `TVT_input ~ (-Z)` の傾き符号で向きを判定。
  - hidden `-Z` min/max を generated `likpf_mean` min/max に対応付け。
  - 最後に `likpf_mean` min/max へ clip。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_typewellanchor_zlikpfminmax_all`
- hidden true `TVT` は guide 生成に使っていない。
- 新規学習、PF/Beam 再生成、提出はなし。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass
- Kaggle package 側 notebook へ copy 済み。

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v9 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 9。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- pull: success。Kaggle source に `typewell tail anchored` / `-Z likPF minmax` / `visual_guides` が反映されていることを確認。
- status: `KernelWorkerStatus.RUNNING`
- output archive は取得していない。
- 2026-07-05 08:00:42 UTC: ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。`ps -ef` で確認したローカル監視プロセスは停止済み。

## v12 ML OOF + direct -Z likPF minmax + exp202 overlay Kaggle v12 実行開始

記録時刻: 2026-07-05 13:38:15 UTC

ユーザー指示:

- 現在の plot に exp202 の結果も含める。

修正:

- exp202 `heatmap_mdn_candidate_generator_probe` の train-side output を読む処理を追加。
- 入力:
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz`
  - `exp202_heatmap_mdn_candidate_generator_probe_candidate_union_by_well.csv`
  - `exp202_heatmap_mdn_candidate_generator_probe_candidate_union_metrics.csv`
- exp202 は sparse topK candidate generator なので、全 row line ではなく TVT panel の candidate point として描画。
- `pred_top1_tvt` は紫の `x` marker、`pred_top2_tvt` から `pred_top10_tvt` は薄い紫 point。
- per-well title / manifest に `existing_oracle_rmse`、`heatmap_union_top10_oracle_rmse`、`new_best_candidate_rate` を追加。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp202_all`
- Kaggle package metadata に `kentookumura/exp202-heatmap-mdn-candgen-train` を kernel source として追加。
- 新規学習、PF/Beam 再生成、hidden inference、提出はなし。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass
- Kaggle package 側 notebook へ copy 済み。package 側に `zlikpfminmax_simple_exp202`、exp202 input resolver、exp202 marker plot、exp202 title metrics が反映されている。

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v12 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 12。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- pull: success。Kaggle source は `exp083-v12-ml-oof-known-tvt-probe.ipynb` として取得された。metadata に exp202 kernel source が反映され、source に `zlikpfminmax_simple_exp202`、`exp202 heatmap top1`、`exp202_candidate_union` が反映されている。
- status: `KernelWorkerStatus.RUNNING`
- 2026-07-05 13:45:49 UTC: ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。`pgrep -af "[k]aggle kernels logs"` と `pgrep -af "[t]imeout 600 kaggle"` でローカル監視プロセスが残っていないことを確認。
- output archive は取得していない。

## v12 ML OOF + direct -Z likPF minmax guide Kaggle v11 実行開始

記録時刻: 2026-07-05 09:17:44 UTC

ユーザー指示:

- plot 対象区間の `Z` をマイナスして `likpf_mean` range に min-max scaling するだけでよい、という方針に差し替える。

修正:

- `-Z likPF minmax anchored` を廃止。
- known tail `TVT_input ~ (-Z)` の向き判定を使わない。
- `last_known_tvt` への anchor shift を使わない。
- 最後の clip を使わない。
- plot 対象 rows の `neg_z = -Z` min/max を source range、同じ rows の generated `likpf_mean` min/max を target range とする。
- `guide = likpf_min + progress(-Z) * (likpf_max - likpf_min)` として `z_likpf_minmax_tvt` に保存。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_all`
- hidden true `TVT` は guide 生成に使っていない。
- 新規学習、PF/Beam 再生成、提出はなし。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass
- Kaggle package 側 notebook へ copy 済み。package 側にも `zlikpfminmax_simple_all`、`direct -Z to likPF range`、`clip_to_range: False` が反映されている。

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v11 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 11。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- pull: success。Kaggle source は `exp083-v12-ml-oof-known-tvt-probe.ipynb` として取得された。`zlikpfminmax_simple_all`、`direct -Z to likPF range`、`plot_neg_z_minmax_to_likelihood_pf_mean_range` が反映されている。
- status: `KernelWorkerStatus.RUNNING`
- 2026-07-05 09:20:21 UTC: ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。`pgrep -af "[k]aggle kernels logs"` と `pgrep -af "[t]imeout 600 kaggle"` でローカル監視プロセスが残っていないことを確認。
- output archive は取得していない。

完了確認:

- 記録時刻: 2026-07-05 09:46:31 UTC
- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` のみ確認。`kaggle kernels output` は実行していない。
- logs: 773/773 plots 作成完了、manifest / plots dir / plots zip / summary の保存先が出力された。
- logs: `Z-to-likPF simple minmax status counts: {'ok': 773}`、coverage min 1.0。
- logs: summary の `visual_guides.z_likpf_minmax_scaling` は `method: plot_neg_z_minmax_to_likelihood_pf_mean_range`、`anchor_used: false`、`known_tail_direction_used: false`、`clip_to_range: false`。
- 現在の `-Z` guide の開始 TVT は明示 anchor では決めていない。plot rows の `-Z` min/max と同じ rows の `likpf_mean` min/max で、最初の plot row の `-Z` の相対位置を `likpf_mean` range に写像した値になる。

## v12 ML OOF + -Z likPF minmax anchored guide Kaggle v10 実行開始

記録時刻: 2026-07-05 08:26:48 UTC

ユーザー指示:

- `typewell tail anchored` は良くなさそうなので削除する。
- `-Z likPF minmax` だけ残して再実行する。
- `-Z likPF minmax` は始まりを `last_known_tvt` にする。

修正:

- typewell guide の描画、typewell CSV 読み込み、typewell guide metadata を削除。
- TVT panel に残す background guide は `-Z likPF minmax anchored` のみ。
- known tail `TVT_input ~ (-Z)` の傾き符号で向きを判定。
- hidden `-Z` min/max を source range、generated `likpf_mean` min/max を target span として min-max scaling。
- scaling 後、最初の finite hidden point が `last_known_tvt` になるように全体を shift。
- clip は `[min(likpf_min, last_known_tvt), max(likpf_max, last_known_tvt)]` で行い、始点 anchor を維持。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_anchor_all`
- hidden true `TVT` は guide 生成に使っていない。
- 新規学習、PF/Beam 再生成、提出はなし。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass
- `rg -ni "typewell_anchor|read_typewell|typewell tail|typewellanchor|anchored progress guide"`: no matches in notebook script.
- Kaggle package 側 notebook へ copy 済み。package 側にも `-Z likPF minmax anchored` と anchor shift が反映され、typewell guide 参照は残っていない。

Kaggle:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v10 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 10。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- pull: success。Kaggle source に `-Z likPF minmax anchored`、anchor shift、`zlikpfminmax_anchor` output prefix が反映されていることを確認。typewell guide 参照はなし。
- status: `KernelWorkerStatus.RUNNING`
- 2026-07-05 08:47:09 UTC: ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。`ps -ef` でローカル監視プロセスが残っていないことを確認。
- output archive は取得していない。

完了確認:

- 記録時刻: 2026-07-05 09:08:26 UTC
- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` のみ確認。`kaggle kernels output` は実行していない。
- logs: 773/773 plots 作成完了、manifest / plots dir / plots zip / summary の保存先が出力された。
- logs: `Z-to-likPF anchored minmax status counts: {'ok': 773}`、coverage min 1.0、direction counts は positive 476 / negative 297。
- logs: `Clip Z guide to range expanded by last known TVT: True`。summary でも `clip_to_likpf_range_expanded_by_anchor: true`。
- 実装上も `CLIP_Z_TO_LIKPF_RANGE` の default は true で、scaled 後の値は `[min(likpf_min, likpf_max, last_known_tvt), max(likpf_min, likpf_max, last_known_tvt)]` に `np.clip` される。始点は clip 後に `last_known_tvt` へ戻している。

## v12 ML OOF + -Z anchor/typewell-tail hybrid Kaggle v7 実行開始

記録時刻: 2026-07-05 06:44:30 UTC

ユーザー確認:

- v6 の known TVT tail fit 版は目視上悪化。
- Kaggle output archive はダウンロードしない。
- v7 では `3. known anchor + typewell tail hybrid` に差し替える。

v6 確認:

- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` のみ確認。`kaggle kernels output` は実行していない。
- v6 logs:
  - output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_ztail50_all`
  - plotted wells: 773/773
  - `Z-to-TVT tail fit status counts: {'ok': 773}`
  - `Z-to-TVT tail fit coverage min: 1.0`
  - `Z-to-TVT known rows min: 851`
  - `Z-to-TVT tail fit used rows min: 50`

v7 修正:

- `-Z known-tail fit` を削除し、`-Z anchor+typewell tail` に差し替え。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zhybrid_typewelltail_all`
- last known `TVT_input` / `Z` を anchor とし、hidden plot rows の `-Z` 95% 点を typewell tail の TVT 95% 点に合わせる affine mapping にした。
- typewell tail は `TVT >= last_known_tvt` を優先し、不足時は typewell `TVT` の 60% quantile 以上へ fallback。
- hidden true `TVT` と raw `TVT` probe は scaling に使わない。
- `TVT_input` prefix rows は引き続き plot frame に追加しない。
- known TVT probe と prediction-start vertical line は引き続き描画しない。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -c "import json; json.load(open('experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb')); print('json ok')"
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass

Kaggle:

- `kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe`
- result: `Kernel version 7 successfully pushed.`
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- `kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v7 -m` で Kaggle 側 source に `zhybrid_typewelltail` / `-Z anchor+typewell tail` が入っていることを確認。
- ユーザー指示により `timeout 600 kaggle kernels logs -f --interval 30 ...` の監視だけ停止。Kaggle 実行自体は停止していない。

## v12 ML OOF + -Z anchor/typewell-tail hybrid Kaggle v7 完了確認

記録時刻: 2026-07-05 07:04:20 UTC

- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` のみ確認。`kaggle kernels output` は実行していない。
- logs:
  - output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zhybrid_typewelltail_all`
  - plotted wells: 773/773
  - `Z-to-TVT hybrid status counts: {'ok': 773}`
  - `Z-to-TVT hybrid coverage min: 1.0`
  - `Z-to-TVT known rows min: 851`
  - `Z-to-TVT typewell tail rows min: 15`
  - `Z-to-TVT typewell tail source counts: {'typewell_after_last_known_tvt': 773}`

ユーザー確認:

- v7 は v6 よりさらに悪化。
- typewell の TVT 範囲を超えてプロットされており、意図通りではない。

実装レビュー:

- 現行 v7 は last known anchor と typewell tail 95% TVT を、hidden `-Z` 95% 点へ affine mapping しているだけで、出力値を typewell TVT 範囲に clip していない。
- hidden `-Z` の 95% 点を超える、または反対方向へ伸びる rows は、typewell tail target より外側へ線形外挿される。
- logs の manifest head でも、`anchor_neg_z` と `hidden_neg_z_target_value` の差が小さい / 符号が逆になり、slope が大きくなる well がある。例: `000d7d20` は `anchor_neg_z=9735.08`、`hidden_neg_z_target_value=9729.4975`、`typewell_tail_target_tvt=11865.250` で `slope=-21.115987`。
- よって「typewell TVT 後半範囲内に収める plot guide」という意図には現行実装は合っていない。修正するなら、anchor から typewell tail high までの clipped progress mapping に変更し、最終値を typewell TVT range に clip する必要がある。

## v12 ML OOF + per-well PF/Beam RMSE title + TVT depth-down Kaggle v4 実行結果

2026-07-05 05:29:30 UTC 確認。

Kaggle kernel:

- `kentookumura/exp083-v12-ml-oof-known-tvt-probe`
- version: v4
- status: `KernelWorkerStatus.COMPLETE`
- output archive: ユーザー指示により未取得。確認は `kaggle kernels status` と `kaggle kernels logs` のみ。

v4 の修正内容:

- 上段 TVT panel の y 軸を `ax.invert_yaxis()` で depth-down 表示に変更。
- RMSE 計算は v3 から変更なし。
- known TVT probe は引き続き非表示。
- prediction-start 縦線は引き続き非表示。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_all`

ログ確認:

- `TVT axis inverted depth-down: True`
- summary `tvt_axis_inverted`: `true`
- `TVT_input` prefix interval plotted: no
- Prediction-start vertical line plotted: no
- Known TVT probe plotted: no
- PF/Beam rows: 3,783,989
- exp148 `lgb_mean` OOF rows: 3,783,989
- Joined rows: 3,783,989
- ML OOF coverage: 1.0
- Plot wells: 773
- Manifest rows: 773
- summary `plot_title_scope`: `per_well`
- summary `plot_title_fields`: `exp148_oof_rmse`, `pfbeam_oracle_rmse`, `pfbeam_best1_rmse`, `pfbeam_best1_label`

ログ上の global reference:

- exp148 OOF RMSE: 8.501290984299567
- PF/Beam oracle RMSE: 6.95303025304006
- PF/Beam best1: `likpf_mean` / `Likelihood PF mean`
- PF/Beam best1 RMSE: 11.594897668440991

## v12 ML OOF + TVT depth-down + -Z known-fit extrap Kaggle v5 実行結果

2026-07-05 05:57:16 UTC 確認。

Kaggle kernel:

- `kentookumura/exp083-v12-ml-oof-known-tvt-probe`
- version: v5
- status: `KernelWorkerStatus.COMPLETE`
- output archive: ユーザー指示により未取得。確認は `kaggle kernels status` と `kaggle kernels logs` のみ。

v5 の修正内容:

- `-Z scaled` を、known `TVT_input` 行で fit した `TVT_input ~ (-Z)` の一次回帰外挿線に変更。
- label: `-Z known-fit extrap`
- fit 対象: raw train well 内の `TVT_input.notna()` 行。
- plot 対象: exp072 feature-cache rows のみ。`TVT_input` prefix rows は plot frame に追加しない。
- TVT depth-down 軸、per-well RMSE title、known TVT probe 非表示、prediction-start 縦線非表示は v4 から継続。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zfit_all`

ログ確認:

- `Z-to-TVT known fit extrap plotted: yes`
- minimum fit points: 8
- PF/Beam rows: 3,783,989
- exp148 `lgb_mean` OOF rows: 3,783,989
- Joined rows: 3,783,989
- ML OOF coverage: 1.0
- Plot wells: 773
- Manifest rows: 773
- `Z-to-TVT fit status counts`: `{'ok': 773}`
- `Z-to-TVT fit coverage min`: 1.0
- `Z-to-TVT fit rows min`: 851
- summary `z_tvt_fit_extrapolation.method`: `linear_regression_tvt_input_vs_negative_z_on_known_rows`
- summary `z_tvt_fit_extrapolation.known_tvt_input_prefix_plotted`: false
- summary `z_tvt_fit_extrapolation.plotted_on`: `exp072_feature_cache_rows_only`

ログ上の global reference:

- exp148 OOF RMSE: 8.501290984299567
- PF/Beam oracle RMSE: 6.95303025304006
- PF/Beam best1: `likpf_mean` / `Likelihood PF mean`
- PF/Beam best1 RMSE: 11.594897668440991

## v12 ML OOF + per-well PF/Beam RMSE title Kaggle v3 実行結果

2026-07-05 04:34:34 UTC 確認。

Kaggle kernel:

- `kentookumura/exp083-v12-ml-oof-known-tvt-probe`
- version: v3
- status: `KernelWorkerStatus.COMPLETE`
- output archive: ユーザー指示により未取得。確認は `kaggle kernels status` と `kaggle kernels logs` のみ。

v3 の修正内容:

- known TVT probe は削除。
- prediction-start 縦線は引き続き非表示。
- plot title は well ごとの指標に変更:
  - `exp148 CV(OOF) RMSE`
  - `PF/Beam oracle RMSE`
  - `PF/Beam best1 RMSE` と best1 label
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_all`

ログ確認:

- `TVT_input` prefix interval plotted: no
- Prediction-start vertical line plotted: no
- Known TVT probe plotted: no
- PF/Beam rows: 3,783,989
- PF/Beam wells: 773
- exp148 `lgb_mean` OOF rows: 3,783,989
- Joined rows: 3,783,989
- ML OOF coverage: 1.0
- Plot wells: 773
- Manifest rows: 773
- summary `plot_title_scope`: `per_well`
- summary `plot_title_fields`: `exp148_oof_rmse`, `pfbeam_oracle_rmse`, `pfbeam_best1_rmse`, `pfbeam_best1_label`

ログ上の global reference:

- exp148 OOF RMSE: 8.501290984299567
- PF/Beam oracle RMSE: 6.95303025304006
- PF/Beam best1: `likpf_mean` / `Likelihood PF mean`
- PF/Beam best1 RMSE: 11.594897668440991

manifest head に per-well 指標列が出力されていることを確認:

- `exp148_oof_rmse`
- `pfbeam_oracle_rmse`
- `pfbeam_best1_column`
- `pfbeam_best1_label`
- `pfbeam_best1_rmse`
- `pf_ancc_rmse`
- `pf_z_rmse`
- `beam_mean_rmse`
- `likpf_mean_rmse`

## v12 ML OOF + known TVT_input extrapolation probe Kaggle v2 実行結果

2026-07-05 04:01:50 UTC 確認。

Kaggle kernel:

- `kentookumura/exp083-v12-ml-oof-known-tvt-probe`
- version: v2
- status: completed by logs / output.
- Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe`

ログ確認:

- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all`
- PF/Beam rows: 3,783,989
- exp148 `lgb_mean` OOF rows: 3,783,989
- joined rows: 3,783,989
- source wells: 773
- plotted wells: 773
- `TVT_input` prefix interval plotted: no
- prediction-start vertical line plotted: no
- known TVT probe fit points: 256
- ML OOF coverage: 1.0
- manifest rows: 773
- summary flags:
  - `tvt_input_prefix_plotted=false`
  - `prediction_start_line_plotted=false`
  - `known_tvt_probe.method=linear_regression_tvt_input_vs_md_tail`
  - `known_tvt_probe.plotted_on=exp072_feature_cache_rows_only`
  - `coverage.manifest_known_tvt_probe_min=1.0`
  - `coverage.manifest_ml_oof_min=1.0`

ローカル output:

- `kaggle/output/v12_ml_oof_known_tvt_probe_v2/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all_plot_manifest.csv`
- `kaggle/output/v12_ml_oof_known_tvt_probe_v2/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all_plots.zip`
- 一部個別 PNG は `kaggle kernels output` の途中停止時点で 461 枚取得済み。全 PNG は zip 内で確認済み。

ローカル検証:

```bash
unzip -l experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v2/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all_plots.zip | tail -5
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
base = Path('experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v2/artifacts')
manifest = base / 'pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all_plot_manifest.csv'
df = pd.read_csv(manifest)
print('rows', len(df))
print('status_counts', df['known_tvt_probe_status'].value_counts(dropna=False).to_dict())
print('known_cov_min', float(df['known_tvt_probe_coverage'].min()))
print('known_fit_rows_min', int(df['known_tvt_probe_fit_rows'].min()))
print('ml_oof_cov_min', float(df['ml_oof_coverage'].min()))
PY
```

結果:

- plots zip: 773 files
- manifest rows: 773
- `known_tvt_probe_status`: `{'ok': 773}`
- `known_tvt_probe_coverage` min: 1.0
- `known_tvt_probe_fit_rows` min: 256
- `ml_oof_coverage` min: 1.0
- 代表画像 `all_wells__000d7d20.png` を目視確認し、青破線 `known TVT_input linear extrap (256)` が表示され、prediction-start 専用縦線は表示されていないことを確認。

未実施:

- Kaggle output 取得と実行完了確認はまだしていない。
- ローカル notebook 実行はルール通り行っていない。

Kaggle 実行開始:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v1 -m
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 1。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- metadata pull: success。`enable_gpu=false`、`enable_internet=false`、kernel sources は `kentookumura/exp072-exp063-full-replay-feature-cache-train` と `kentookumura/exp148-train`、competition source は `rogii-wellbore-geology-prediction`。
- `logs -f` は CLI 側でログ本文なしのまま監視中だったため、ユーザー指示により監視だけ停止した。Kaggle 側実行は停止していない。

Kaggle 実行完了確認:

```bash
kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe
kaggle kernels output kentookumura/exp083-v12-ml-oof-known-tvt-probe -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v1
unzip -t experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots.zip
unzip -q -o experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots.zip -d experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots
```

結果:

- status: completed by logs / output.
- Kaggle runtime: CPU, internet disabled.
- rows: PF/Beam 3,783,989、exp148 `lgb_mean` OOF 3,783,989、joined 3,783,989。
- wells: source 773、plotted 773。
- plot scope: all wells。
- `TVT_input` prefix plotted: false。
- ML OOF coverage: 1.0。
- raw TVT probe coverage: 1.0。
- source SHA:
  - exp072 PF/Beam gzip `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - exp072 PF/Beam decompressed `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
  - exp148 OOF gzip `12f2980972c19ef72a88b198efa0f5329ee3614a21b269f1bebc5a37b3ac21b5`
  - exp148 OOF decompressed `ec28d89641b74c67482aff7a1ebc925db536716f1a024467ae0339dd2326e14d`
- local output:
  - `kaggle/output/v12_ml_oof_known_tvt_probe_v1/exp083-v12-ml-oof-known-tvt-probe.log`
  - `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plot_manifest.csv`
  - `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots.zip`
  - `kaggle/output/v12_ml_oof_known_tvt_probe_v1/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_probe_all_plots/`
- manifest rows: 773、unique wells: 773、manifest min `raw_tvt_coverage=1.0`、manifest min `ml_oof_coverage=1.0`。
- plots zip: 255M、zip PNG count 773、`unzip -t` PASS。
- extracted local PNG count: 773。

補足:

- `kaggle kernels output` は個別 PNG 取得が遅いため途中停止したが、manifest と plots zip は取得済み。検証済み zip からローカル PNG ディレクトリを再作成した。
- `--file-pattern '.*summary.*'` は CLI page 都合で log のみ取得したため、summary JSON 本体はローカル未取得。ただし同内容は logs に出力され、manifest / zip で実ファイル整合を確認済み。

表示改善:

- ユーザー確認により、v1 の `known raw TVT probe` が plot 上で見えないことを確認。
- 原因は raw train `TVT` probe が exp072 復元 `true_tvt` と同じ座標にあり、さらに v1 実装では青い小点を先に描いた後で黒い `true TVT` 線を太く描いていたため、黒線の下に隠れていたこと。
- notebook script と ipynb を修正し、`true TVT` 黒線の後に `known raw TVT probe` を青い中抜き丸 (`facecolors=none`, `edgecolors=#2563eb`, `s=16`, `zorder=7`) として描くようにした。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。
- 検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_v12_ml_oof_known_tvt_probe_jsoncheck.out
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass
- 修正版の Kaggle v2 実行はまだしていない。

## v12 ML OOF + known TVT_input extrapolation probe 修正

ユーザー確認により、`known raw TVT probe` の意図は raw train `TVT` を tail rows に重ねることではなく、known `TVT_input` 区間の最後 N 点を回帰して tail/evaluation rows へ外挿することだと判明した。また prediction start の縦線は不要との指示。

修正:

- `known raw TVT probe` を削除。
- raw well の `TVT_input.notna()` 末尾 `KNOWN_TVT_PROBE_POINTS=256` 点で `TVT_input ~ MD` の一次回帰を行い、exp072 feature-cache rows の raw `MD` へ外挿する。
- plot label は `known TVT_input linear extrap (256)`。
- known `TVT_input` prefix rows は plot frame に追加せず、tail/evaluation rows だけに外挿線を描く。
- prediction start の縦線と関連関数を削除。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_known_tvt_extrap_probe_all` に変更し、v1 の raw TVT probe output と混ざらないようにした。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/python -m json.tool experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb >/tmp/exp083_v12_ml_oof_known_tvt_probe_jsoncheck.out
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- `.ipynb` JSON check: pass

## Latest state: v12 ML OOF + exp209 HMM +/-2sigma band

ユーザー指示により、HMM の 2sigma range を薄い band として TVT panel に追加した。直前の v24 no-formation 版は Kaggle status/logs で COMPLETE を確認済み。output は取得していない。

修正:

- exp209 HMM mean の周りに `hmm_mean_tvt +/- 2*hmm_std` を薄い紫色の `fill_between` band として描画する。
- band は `hmm_std` が finite かつ 0 以上の点だけを使い、同じ `md_since` は median 集約する。
- HMM mean 線は従来通り残し、`likPF/HMM blend` は引き続き描画しない。
- HMM +/-2sigma band の min/max も TVT panel の y-axis range に含める。
- manifest / summary に `exp209_hmm_2sigma_*` と band formula を追加した。
- 地層 line / band は引き続き描画しない。
- PNG 保存名は引き続き `{well}.png` で、`all_wells__` prefix は使わない。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all` に変更した。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も更新済み。

検証:

- `py_compile`: pass
- `ruff --select F821`: pass
- source `.py` ASCII check: pass
- `jupytext --to ipynb`: pass
- `jupytext --to ipynb --test`: pass
- local/package `.ipynb` JSON check: pass
- Package notebook grep: `hmm_2sigma` / `+/-2sigma` / `2sigma_noformation` あり、`all_wells__` / `blend_likpf` / `formation_axis_context` はなし。
- Kaggle に v25 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。
- 実行中 logs は空。
- ユーザー指示により監視だけ停止した。Kaggle 側実行は継続。

v25 完了確認:

- ユーザーから完了連絡あり。
- `kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe`: `KernelWorkerStatus.COMPLETE`。
- `kaggle kernels logs kentookumura/exp083-v12-ml-oof-known-tvt-probe` で logs を確認した。
- output prefix: `pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all`
- PF/Beam rows: 3,783,989。
- exp148 `lgb_mean` OOF rows: 3,783,989。
- exp209 enriched HMM rows: 3,783,989。
- joined rows: 3,783,989。
- source wells: 773、plotted wells: 773。
- manifest rows: 773。
- `Z-to-likPF simple minmax status counts`: `{'ok': 773}`。
- `Z-to-likPF simple minmax coverage min`: 1.0。
- exp209 HMM rows per well min/max: 407 / 10052。
- exp209 HMM mean points per well min/max: 407 / 10052。
- exp209 HMM mean TVT range: 10047.468 / 12888.823。
- exp209 HMM +/-2sigma points per well min/max: 407 / 10052。
- exp209 HMM +/-2sigma TVT range: 10027.519931 / 12892.8001084。
- summary coverage `manifest_exp209_hmm_2sigma_points_min`: 407。
- summary outputs:
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_plot_manifest.csv`
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_plots.zip`
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_summary.json`
- output は取得していない。

## Latest state: v12 ML OOF + exp209 HMM +/-2sigma + exp226 K16 OOF overlay

ユーザー指示により、exp226 `connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` の train OOF 予測も TVT panel に重ねて表示するようにした。新規学習、PF/Beam 再生成、推論、提出は行っていない。

修正:

- exp226 train OOF `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz` を入力に追加。
- Kaggle input hint / package metadata に `kentookumura/exp226-k16-kappa-repro-train` を追加。
- exp226 OOF の `well_id,row_idx,tvt_pred` を `id,well,exp226_k16_oof_tvt` に変換し、exp072 feature-cache plot frame に `id,well` で結合。
- TVT panel に `exp226 K16 OOF` 線を追加。
- plot title に per-well `exp226_oof_rmse` を追加。
- manifest に `exp226_oof_coverage` を追加。
- summary に exp226 recorded CV RMSE、source path、gzip SHA、decompressed SHA、rows、coverage、global/per-well RMSE field を追加。
- output prefix を `pf_beam_true_tvt_2d_well_eda_v12_exp148_exp226_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all` に変更し、直前 v25 output と混ざらないようにした。
- Kaggle package 側 `kaggle/v12_ml_oof_known_tvt_probe/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.ipynb` も同期済み。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
.venv/bin/ruff check experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --set-kernel python3 experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp083_pf_beam_true_tvt_2d_well_eda/exp083_pf_beam_true_tvt_2d_well_eda_v12_ml_oof_known_tvt_probe.py
```

- `py_compile`: pass
- `ruff --select F821`: pass
- `jupytext --to ipynb --set-kernel python3`: pass
- `jupytext --to ipynb --test`: pass
- local notebook / package notebook / package `kernel-metadata.json` JSON parse: pass
- local notebook と package notebook の `cmp -s`: pass
- `rg` で local source、package notebook、metadata に `exp226` と `kentookumura/exp226-k16-kappa-repro-train` が入っていることを確認。

Kaggle v26 push / 完了確認:

```bash
kaggle kernels push -p experiments/exp083_pf_beam_true_tvt_2d_well_eda/kaggle/v12_ml_oof_known_tvt_probe
kaggle kernels pull kentookumura/exp083-v12-ml-oof-known-tvt-probe -p /tmp/kaggle-pull/exp083-v12-ml-oof-known-tvt-probe-v26 -m
kaggle kernels status kentookumura/exp083-v12-ml-oof-known-tvt-probe
timeout 300 kaggle kernels logs -f --interval 30 kentookumura/exp083-v12-ml-oof-known-tvt-probe
```

- push: success。Kernel version 26。
- URL: https://www.kaggle.com/code/kentookumura/exp083-v12-ml-oof-known-tvt-probe
- pull: success。Kaggle source / metadata に `exp226` と `kentookumura/exp226-k16-kappa-repro-train` が反映されていることを確認。
- final status: `KernelWorkerStatus.COMPLETE`。
- PF/Beam rows: 3,783,989。
- exp148 `lgb_mean` OOF rows: 3,783,989。
- exp226 K16 OOF rows: 3,783,989。
- exp209 enriched HMM rows: 3,783,989。
- joined rows: 3,783,989。
- source wells: 773、plotted wells: 773。
- global exp148 OOF RMSE: 8.50。
- global exp226 OOF RMSE: 9.43。
- global PF/Beam oracle RMSE: 6.95。
- global PF/Beam best1 RMSE: 11.59 (`Likelihood PF mean`)。
- exp226 OOF coverage: 1.0、manifest min/max: 1.0 / 1.0。
- manifest rows: 773。
- `Z-to-likPF simple minmax status counts`: `{'ok': 773}`。
- `Z-to-likPF simple minmax coverage min`: 1.0。
- exp209 HMM rows per well min/max: 407 / 10052。
- exp209 HMM mean points per well min/max: 407 / 10052。
- exp209 HMM mean TVT range: 10047.468 / 12888.823。
- exp209 HMM +/-2sigma points per well min/max: 407 / 10052。
- exp209 HMM +/-2sigma TVT range: 10027.519931 / 12892.8001084。
- summary source SHA:
  - exp226 OOF decompressed: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
  - exp226 OOF gzip: `4151b8fecd0caf1cdb58ef00c565162dff89a5868b4735639d51baa83cebd134`
- summary outputs:
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_exp226_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_plot_manifest.csv`
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_exp226_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_plots.zip`
  - `/kaggle/working/artifacts/pf_beam_true_tvt_2d_well_eda_v12_exp148_exp226_oof_pfbeam_rmse_title_tvt_down_zlikpfminmax_simple_exp209_hmm_2sigma_noformation_all_summary.json`
- output archive は取得していない。
