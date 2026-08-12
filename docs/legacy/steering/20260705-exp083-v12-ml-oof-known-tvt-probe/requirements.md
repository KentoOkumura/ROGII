# 要件

## 依頼

exp083 の v12 prediction-start plot を拡張した別 notebook を作成する。既存の正規 train notebook は上書きしない。

- exp148 `lgb_mean` OOF prediction を読み、ML 予測値として well plot に重ねる。
- train raw の known `TVT` を tail/evaluation rows の probe として plot し、exp072 `true_tvt` 復元との alignment を確認できるようにする。
- notebook は EDA / visualization 専用で、学習、候補生成、提出は行わない。

## 制約

- Route: `pf_beam` 主体の診断 notebook。exp148 ML OOF は比較用 overlay であり、新しい ensemble policy ではない。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp148 OOF は保存済み `exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz` を読むだけにし、exp148 control / baseline は再学習しない。
- `TVT_input` の既知 prefix 区間はプロット対象外とし、plot range は exp072 feature cache rows に限定する。
- known raw `TVT` probe は train-side 可視化専用で、inference rule、hard router、submission には使わない。
- 既存 `exp083_pf_beam_true_tvt_2d_well_eda_train.ipynb` は変更しない。

## 受け入れ基準

- `experiments/exp083_pf_beam_true_tvt_2d_well_eda/` に別名の Jupytext percent `.py` と `.ipynb` が作成されている。
- notebook 上で入力パス、exp148 OOF model/filter、selected wells、生成物パスが確認できる。
- 全 773 well をプロット対象にする。
- 各 plot は少なくとも true TVT、PF Z、PF ANCC、Beam mean、Likelihood PF mean、exp148 ML OOF、known raw TVT probe、prediction start line を表示できる。
- `TVT_input` prefix rows は追加されていない。
- 構文チェック、`jupytext --to ipynb --test`、`ruff --select F821` が通る。
- この notebook は diagnostic visualization であり deterministic prediction anchor ではない、と記録されている。

## 2026-07-05 follow-up

- 既存の `-Z known-fit extrap` は、known `TVT_input` 全区間 fit ではなく known 区間末尾の tail fit に差し替える。
- tail fit は hidden/evaluation rows の真の `TVT` を使わず、known `TVT_input` と raw `Z` のみを使う。
- `TVT_input` prefix rows 自体は引き続きプロット対象外とし、fit の入力にだけ使う。
- prediction start line と known TVT probe は引き続き描画しない。

## 2026-07-05 follow-up 2

- known tail fit 版は目視上悪化したため、`-Z` guide を known anchor + typewell tail hybrid に差し替える。
- last known `TVT_input` / `Z` を anchor とし、hidden plot rows の `-Z` 上位 quantile を typewell 後半 TVT quantile へ合わせる。
- typewell の `TVT` は提供データだけを使い、hidden/evaluation rows の真の `TVT` は scaling に使わない。

## 2026-07-05 follow-up 3

- anchor affine 版は typewell TVT 範囲を超えるため廃止する。
- `-Z` guide は known tail の `TVT_input` vs `-Z` の向きだけを判定に使う。
- hidden plot rows の `-Z` min/max を typewell tail TVT min/max に min-max scaling する。
- 向きが正なら hidden `-Z` min -> typewell tail TVT min、hidden `-Z` max -> typewell tail TVT max。
- 向きが負なら hidden `-Z` min -> typewell tail TVT max、hidden `-Z` max -> typewell tail TVT min。
- 最後に typewell tail TVT min/max へ clip する。

## 2026-07-05 follow-up 4

- feature candidate として見たときの挙動確認のため、以下 2 本を TVT panel に可視化する。
- 1 本目は last known `TVT_input` を anchor にし、typewell の `TVT >= last_known_tvt` tail を hidden 区間 progress 0..1 で補間する。
- typewell guide は `guide = last_known_tvt + (typewell_tail_interp - typewell_tail_start)` とし、hidden/evaluation rows の真の `TVT` は使わない。
- 2 本目は hidden `-Z` min/max を source range、生成済み `Likelihood PF mean` min/max を target range として min-max scaling する。
- `-Z` の向きは known tail の `TVT_input ~ (-Z)` 傾き符号だけで判定し、最後に `Likelihood PF mean` range へ clip する。
- 旧 `-Z typewell-tail minmax` は今回の比較対象から外す。

## 2026-07-05 follow-up 5

- `typewell tail anchored` は目視上良くなさそうなので削除する。
- TVT panel に残す guide は `-Z likPF minmax` のみとする。
- `-Z likPF minmax` は generated `Likelihood PF mean` min/max の幅を使うが、最初の hidden row が `last_known_tvt` になるように全体を shift する。
- shift 後の clip は generated `Likelihood PF mean` min/max に `last_known_tvt` を含めた range で行い、anchor 始点を壊さない。

## 2026-07-05 follow-up 6

- v10 の `-Z likPF minmax anchored` は clip による上下張り付きが見えるため廃止する。
- TVT panel に残す guide は引き続き `-Z likPF minmax` のみとする。
- plot 対象区間の raw `Z` を `-Z` に変換し、その min/max を source range とする。
- 同じ plot 対象区間の generated `Likelihood PF mean` min/max を target range とする。
- `guide = likpf_min + progress(-Z) * (likpf_max - likpf_min)` の単純 min-max scaling とし、known tail direction、`last_known_tvt` anchor shift、final clip は使わない。
- hidden/evaluation rows の真の `TVT` は使わない。

## 2026-07-05 follow-up 7

- 既存 plot に exp202 `heatmap_mdn_candidate_generator_probe` の train-side 結果も重ねる。
- exp202 は direct TVT prediction ではなく sparse topK candidate generator なので、全 row line ではなく candidate point として描画する。
- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz` の top1 を目立つ marker、top2-10 を薄い marker として TVT panel に重ねる。
- `candidate_union_by_well.csv` の `existing_oracle_rmse`、`heatmap_union_top10_oracle_rmse`、`new_best_candidate_rate` を well title / manifest に追加する。
- exp202 の hidden test inference、submission、PF/Beam 再生成、新規学習は行わない。

## 2026-07-05 follow-up 8

- exp202 overlay の `pred_top1_tvt` は、sample center の順序に沿って線としてプロットする。
- 線の x 軸は exp072 feature-cache rows に join した `md_since` とし、well ごとに `md_since` 昇順で接続する。
- `pred_top2_tvt` から `pred_top10_tvt` は引き続き薄い sparse point として残す。
- exp202 の保存済み output には window 内 `pred_path_tvt` 配列は含まれないため、今回描くのは strict full path ではなく top1 center path であることを summary に明記する。
- exp202 の hidden test inference、submission、PF/Beam 再生成、新規学習は行わない。

## 2026-07-06 follow-up 9

- exp202 v2 で保存された path artifact に合わせ、plot notebook は `pred_top1_tvt` center line ではなく saved local path segments を描画する。
- 追加で読む exp202 artifact は以下とする。
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
  - `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`
- `pred_tvt_path[sample, rank, step]` を TVT panel に重ね、rank1 を濃い線、rank2-10 を薄い線として表示する。
- path の x 軸は NPZ の absolute `md_path` を sample center の `md_since_prefix` に合わせて `md_since` に変換する。
- path segment は plot 対象区間内だけに clip し、`TVT_input` prefix rows は引き続き plot 対象外とする。
- exp202 path は local 128-row window segment であり、full-well trajectory stitching ではないことを summary に明記する。
- exp202 の hidden test inference、submission、PF/Beam 再生成、新規学習は行わない。

## 2026-07-06 follow-up 10

- v14 の saved local path segment overlay は短い segment 群として見え、期待する path 表示ではないため差し替える。
- exp202 artifact 自体は full-well trajectory ではなく local 128-row windows である前提は維持する。
- plot では local window 点を global `md_since` 座標へ戻し、rank1 の予測 TVT を x ごとに median 集約して stitched guide として描画する。
- x 座標は `horizontal_row_index - prefix_end` を使い、`TVT_input` prefix rows は引き続き plot 対象外とする。
- rank2-10 segment と center marker は描画せず、plot を rank1 stitched guide に絞る。
- summary には stitched guide が可視化用集約であり、exp202 の direct evaluation score や native full-well trajectory ではないことを明記する。

## 2026-07-06 follow-up 11

- v16 の rank1 stitched guide は正常完了したが、full local path logits の center 以外の step が大きく飛び、紫線だけ TVT panel の scale を壊すため廃止する。
- exp202 の path artifact から使う描画対象は `candidate_path_rank_index.csv.gz` の rank1 `center_pred_tvt` に限定する。
- `candidate_path_samples.csv.gz` の `md_since_prefix` と `path_npz_sample_index` で rank1 center prediction を sample center 順に接続する。
- full `pred_tvt_path` の全 step stitching は描画しない。
- TVT panel の y 軸範囲は true/PF/Beam/ML/-Z guide から決め、exp202 の out-of-range center candidate が autoscale を壊さないようにする。
- manifest / summary には rank1 center path point 数と TVT min/max を出し、scale 異常を logs で確認できるようにする。

## 2026-07-06 follow-up 12

- exp202 rank1 center path overlay を exp207 `heatmap_mdn_overlapping_window_path_stitch_probe` の output に差し替える。
- 追加で読む exp207 artifact は以下とする。
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_stitched_path_rows.csv.gz`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_by_well.csv`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_metrics.csv`
  - `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_source_window_coverage.csv`
- `stitched_path_rows.csv.gz` の `stitched_candidate_rank` 1-3 を TVT panel に描画し、rank1 を濃い紫線、rank2-3 を薄い紫線にする。
- x 軸は exp072 feature-cache rows へ `id/well` join した `md_since` を使い、`TVT_input` prefix rows は引き続き plot 対象外とする。
- TVT panel の y 軸範囲は true/PF/Beam/ML/-Z guide から決め、exp207 の out-of-range stitched point が autoscale を壊さないようにする。
- title / manifest / summary は exp207 covered-row candidate-union oracle readout を表示する。これは direct prediction score ではないことを明記する。

## 2026-07-07 follow-up 13

- exp207 overlay を exp210 `heatmap_mdn_full_well_path_generation_probe` の full-well candidate path output に差し替える。
- 使用する primary artifact は `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz` とする。
- `path_rank` 1-5 を TVT panel に描画し、rank1 を濃い紫線、rank2-5 を薄い紫線にする。
- x 軸は exp210 artifact の `md_from_ps`、y 軸は `tvt_pred` を直接使う。exp072 feature-cache rows への `id/well` join は行わない。
- Kaggle package metadata の kernel source は `kentookumura/exp210-hmdn-full-well-path-generation-train` に差し替える。
- title / manifest / summary は exp210 covered-row candidate-union oracle readout を表示する。これは direct prediction score ではないことを明記する。

## 2026-07-07 follow-up 14

- exp210 overlay を exp212 `heatmap_mdn_full_grid_path_generation_probe` の full-grid candidate path output に差し替える。
- 使用する primary artifact は `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz` とする。
- `path_rank` 1-5 を TVT panel に描画し、rank1 を濃い紫線、rank2-5 を薄い紫線にする。
- x 軸は exp212 artifact の `md_since`、y 軸は `tvt_pred` を直接使う。`md_from_ps` は source coverage 記録用に保持する。
- exp212 rank1 は y 範囲で落とさず、rank1 min/max を TVT panel の表示範囲に含める。
- manifest には rank1 の source/fallback point 数、fallback rate、coverage rate、表示 y 範囲を記録する。
- Kaggle package metadata の kernel source は `kentookumura/exp212-hmdn-full-grid-path-generation-train` に差し替える。
- title / manifest / summary は exp212 covered-row candidate-union oracle readout を表示する。これは direct prediction score ではないことを明記する。

## 2026-07-07 follow-up 15

- exp212 overlay を exp215 `mtp_full_tail_heatmap_path_generator_probe` の full-grid candidate path output に差し替える。
- 使用する primary artifact は `exp215_mtp_full_tail_heatmap_path_generator_probe_full_grid_candidate_paths.csv.gz` とする。
- `path_rank` 1-5 を TVT panel に描画し、rank1 を濃い紫線、rank2-5 を薄い紫線にする。
- `weighted_tvt_pred` が保存されている rank1 rows から、learned MTP weighted path をオレンジ破線として追加表示する。
- x 軸は exp215 artifact の `md_since`、y 軸は `tvt_pred` / `weighted_tvt_pred` を直接使う。
- exp215 rank1 と weighted path は y 範囲で落とさず、TVT panel の表示範囲に含める。
- manifest には rank1 source/fallback point 数、fallback rate、coverage rate、path probability / cost / source window count 平均、weighted path の点数と TVT 範囲を記録する。
- Kaggle package metadata の kernel source は `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train` に差し替える。
- title / manifest / summary は exp215 `existing_plus_learned_mtp_topk` と `existing_union` の covered-row candidate-union oracle readout を表示する。これは direct prediction score ではないことを明記する。

## 2026-07-07 follow-up 16

- exp215 output は描画対象から外す。
- exp215 kernel source も plot notebook の input source から外す。
- 代わりに exp209 `exp209-joint-exact-parity-train` の HMM output を TVT panel に重ねる。
- 使用する primary artifact は `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz` とする。
- `hmm_mean_tvt` を HMM mean 線として描画する。
- exp209 の best candidate である `blend_likpf_hmm_w500` も比較線として描画する。
- x 軸は exp072 plot frame に `id/well` join した `md_since`、y 軸は `hmm_mean_tvt` / `blend_likpf_hmm_w500` を直接使う。
- title / manifest / summary は `exp209_vs_exp072_exp205_by_well_delta.csv` の well ごとの HMM RMSE、best blend RMSE、likPF RMSE を表示する。
- summary には `exp209_vs_exp072_exp205_overall_metrics.csv` と `exp209_vs_exp072_exp205_summary.json` を記録する。

## 2026-07-07 follow-up 17

- 背景に薄く表示する地層境界は、plot y-range への共通 min-max rescale を行わず、raw TVT/depth scale のまま描画する。
- 地層境界のラベルから `scaled` 表記を外す。
- exp209 overlay から `likPF/HMM blend` 線を削除し、TVT panel に描画する exp209 系は `hmm_mean_tvt` のみとする。
- title / manifest / summary から blend RMSE、blend point 数、blend TVT range を外す。
- title には exp209 `hmm_mean_tvt` RMSE と比較用の exp072 likPF RMSE だけを残す。
- exp209 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 18

- v22 では地層境界が画面外に出て表示されなくなったため、formation 列を raw 値のまま TVT 軸に描く実装を廃止する。
- formation 列 `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` は `Z` と同じ座標系の境界値として扱い、`formation_tvt = raw_TVT + raw_Z - formation_Z` で TVT 軸へ変換して描画する。
- 地層背景は common min-max rescale ではなく、物理的な Z-to-TVT 差分変換で表示する。
- TVT y-axis には予測値範囲を挟む近傍の formation bracket だけを含め、全地層を入れて予測線が過度に潰れないようにする。
- `likPF/HMM blend` は引き続き描画対象外とする。

## 2026-07-07 follow-up 19

- 地層境界および地層 band は描画対象から完全に外す。
- `-Z likPF minmax` guide は地層ではないため引き続き残す。
- 地層を y-axis range driver にも使わない。
- manifest / summary から地層 context 指標を削除する。
- PNG ファイル名から `all_wells__` prefix を削除し、`{well}.png` として保存する。
- `likPF/HMM blend` は引き続き描画対象外とする。

## 2026-07-07 follow-up 20

- exp209 HMM mean の周りに `hmm_mean_tvt +/- 2*hmm_std` の band を薄い色で描画する。
- band は `hmm_std` が finite かつ 0 以上の点だけを使い、同じ `md_since` は median 集約してから `fill_between` で描画する。
- HMM mean 線は従来通り残し、`likPF/HMM blend` は引き続き描画しない。
- HMM +/-2sigma band の min/max も TVT panel の y-axis range に含める。
- manifest / summary に band の point 数、TVT min/max、計算式を記録する。
- band は posterior uncertainty の可視化 guide であり、calibrated 95% interval としては扱わない。
