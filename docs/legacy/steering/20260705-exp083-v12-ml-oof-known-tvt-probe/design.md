# 設計

## アプローチ

exp083 の既存 helper を丸ごと呼ぶのではなく、v12 plot に必要な path 解決、feature cache 読み込み、raw train context join、exp148 OOF join、plot 保存だけを別名 Jupytext notebook に実装する。正規 train notebook と `config.yaml` は上書きしない。

OOF は `model == "lgb_mean"`、`variant == learned_likelihood_confidence_addonly`、`mode == gpu_repro_guard_dp_threads8` を優先して filter する。local artifact がなければ Kaggle input `/kaggle/input/**/exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz` から探す。

known TVT probe は raw train horizontal CSV の `TVT` を exp072 feature cache の `id={well}_{row_idx}` から復元した raw row index で join し、feature cache rows 上にだけ表示する。`TVT_input` の既知 prefix rows は plot frame に追加しない。v12 と同じく prediction start は `md_since=0`、範囲外の場合は先頭 x へ fallback する。

## 実験範囲

- 対象実験: `exp083_pf_beam_true_tvt_2d_well_eda`
- Route: `pf_beam` 診断。exp148 OOF は ML overlay であり route anchor の更新はしない。
- 親実験: exp072 full replay feature cache、比較 overlay として exp148 OOF。
- 変更する変数: 別名 visualization notebook とその生成 plot prefix。plot 対象は全 well。
- 固定する変数: exp072 feature cache、exp148 saved OOF、raw train physical/TVT context、PF/Beam candidates。

## 再現性設計

- seed policy: RNG は使わない。selected wells は明示リスト、または deterministic な sort による上位抽出のみ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存済み exp072 candidate columns を読むだけ。
- 並列処理と乱数の関係: 並列処理なし。
- CPU/GPU runtime と deterministic flags: CPU only の可視化 notebook。GPU 学習なし。
- train cache / test feature regeneration の SHA 記録方針: notebook summary に source file path、row count、well count、必要なら gzip/decompressed SHA を出す。提出候補ではないため anchor SHA ではない。
- model manifest / prediction / submission SHA 記録方針: exp148 OOF の `lgb_mean` prediction SHA は exp148 metrics の既存値を参照し、notebook では読み込んだ OOF の row count と filter 条件を表示する。submission は生成しない。
- Kaggle package bootstrap 確認方針: 既存 prepare flow ではなく別 notebook を直接生成する。Kaggle 実行する場合は exp072 train output と exp148 train output を input source に追加する。

## リスク

- リークリスク: raw `TVT` probe と exp148 OOF は train-side 診断限定。直接 inference rule に使うと leakage になるため、plot title と summary に diagnostic-only と明記する。`TVT_input` prefix 区間は plot 対象外にして、v12 tail/evaluation rows だけを可視化する。
- CV/LB 不一致リスク: plot は目視診断であり、CV/LB を主張しない。
- ランタイム/メモリリスク: exp148 OOF は全 model 分を読むと約 4 倍 rows になるため、読み込み直後に `lgb_mean` に filter する。plot は全 773 well なので PNG/zip 書き出し時間に注意する。
- 再現性リスク: local と Kaggle input の path 差分が主なリスク。candidate path を notebook 上で全て表示して fail-fast する。

## 2026-07-05 follow-up design

`-Z` guide は `TVT_input ~ (-Z)` を known 区間全体ではなく known 末尾 `Z_TVT_TAIL_FIT_POINTS` 点で fit する。デフォルトは 50 点とし、Kaggle 上では環境変数で変更可能にする。fit 後の値は TVT 座標として描画し、formation columns の quantile scaling とは分離する。

fit metadata として、tail points、known rows、slope、intercept、hidden plot rows coverage を manifest と summary に保存する。known prefix は fit 入力にだけ使い、plot frame には追加しない。

## 2026-07-05 follow-up 2 design

known tail fit の傾きが局所的すぎて目視スケールが悪化したため、`-Z` guide は anchor + typewell tail hybrid にする。

手順:

- `last_known_tvt = known.TVT_input.iloc[-1]`、`anchor_neg_z = -known.Z.iloc[-1]` を取得する。
- typewell `TVT` から `last_known_tvt` 以降の行を優先し、不足する場合は typewell 全体の後半 quantile 以上を使う。
- hidden plot rows の `-Z` 95% 点を、typewell tail の 95% TVT 点へ affine mapping する。
- affine line は必ず last known anchor を通す。

この guide は plot 用の座標補助であり、hidden true TVT や raw `TVT` probe は scaling に使わない。

## 2026-07-05 follow-up 3 design

anchor + typewell tail の affine mapping は typewell TVT 範囲外へ外挿されるため、min-max scaling に置き換える。

手順:

- known tail `Z_TVT_DIRECTION_TAIL_POINTS` 点で `TVT_input ~ (-Z)` の一次傾きを計算し、符号だけを使う。
- typewell tail は `TVT >= last_known_tvt` を優先し、不足する場合は typewell 全体の `TYPEWELL_TAIL_START_QUANTILE` 以上を使う。
- hidden plot rows の `-Z` min/max を source range、typewell tail TVT min/max を target range とする。
- slope sign が正なら source min/max を target min/max に対応させ、負なら source min/max を target max/min に対応させる。
- 生成値は最後に target min/max へ clip する。

この guide は plot 用で、hidden true TVT、raw train `TVT`、PF/Beam/ML 予測値は scaling に使わない。

## 2026-07-05 follow-up 4 design

feature candidate の事前確認として、TVT-scale overlay を 2 本に絞る。

1. typewell anchored progress guide
   - raw well の known `TVT_input` 末尾値を `last_known_tvt` とする。
   - typewell `TVT` のうち `TVT >= last_known_tvt` を tail とし、不足時だけ後半 quantile へ fallback する。
   - hidden plot rows の `md_since` を 0..1 progress に正規化し、typewell tail を progress 上で線形補間する。
   - `guide = last_known_tvt + (typewell_tail_interp - typewell_tail_start)` として anchor に接続する。

2. `-Z` to generated Likelihood PF mean min-max guide
   - known tail `TVT_input ~ (-Z)` の傾き符号で向きを判定する。
   - hidden plot rows の `-Z` min/max を source range とする。
   - 同じ rows の生成済み `likpf_mean` min/max を target range とする。
   - 向きが正なら source min/max を target min/max に、負なら source min/max を target max/min に対応させる。
   - 最後に `likpf_mean` min/max へ clip する。

どちらも plot 用 diagnostic であり、hidden true `TVT` は使わない。`Likelihood PF mean` は既存の生成済み PF/Beam candidate で、今回新規生成や学習はしない。

## 2026-07-05 follow-up 5 design

typewell anchored progress guide は削除し、`-Z` to generated Likelihood PF mean min-max guide だけを残す。

手順:

- known tail `TVT_input ~ (-Z)` の傾き符号で向きを判定する。
- hidden plot rows の `-Z` min/max を source range とする。
- 同じ rows の生成済み `likpf_mean` min/max を target span とする。
- 向きに応じて `-Z` を `likpf_mean` min/max range へ min-max scaling する。
- その後、最初の finite hidden point が `last_known_tvt` になるように全体を shift する。
- shift 後の clip は `[min(likpf_min, last_known_tvt), max(likpf_max, last_known_tvt)]` で行い、始点 anchor を維持する。

hidden true `TVT` と typewell `TVT` はこの guide 生成に使わない。plot 用 diagnostic であり、feature 化する場合は別実験で OOF ablation する。

## 2026-07-05 follow-up 6 design

v10 の anchor shift + clip 版は上下張り付きが出るため、`-Z` guide を単純 min-max scaling に戻す。

手順:

- plot 対象 rows の raw `Z` から `neg_z = -Z` を作る。
- `neg_z_min`、`neg_z_max` を source range とする。
- 同じ rows の generated `Likelihood PF mean` から `likpf_min`、`likpf_max` を target range とする。
- `progress = (neg_z - neg_z_min) / (neg_z_max - neg_z_min)` を計算する。
- `guide = likpf_min + progress * (likpf_max - likpf_min)` として TVT panel に重ねる。

known `TVT_input`、known tail direction、`last_known_tvt` anchor shift、typewell、hidden true `TVT`、final clip は使わない。plot 用 diagnostic であり、feature 化する場合は別実験で OOF ablation する。

## 2026-07-05 follow-up 7 design

exp202 の heatmap-MDN 候補生成器結果を、既存 TVT panel に sparse candidate overlay として追加する。

入力:

- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz`
- `exp202_heatmap_mdn_candidate_generator_probe_candidate_union_by_well.csv`
- `exp202_heatmap_mdn_candidate_generator_probe_candidate_union_metrics.csv`

Kaggle では `kentookumura/exp202-heatmap-mdn-candgen-train` を kernel source に追加する。local では既存の `experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v1/artifacts/` を読む。

描画:

- `heatmap_candidates` は `id` / `well` で exp072 feature-cache rows に join し、`md_since` を得る。
- `pred_top1_tvt` は紫の `x` marker として描画する。
- `pred_top2_tvt` から `pred_top10_tvt` は薄い紫 point として描画する。
- sparse candidate なので線で結ばない。

タイトル / manifest:

- `candidate_union_by_well.csv` から per-well の existing oracle RMSE、existing+heatmap top10 oracle RMSE、new-best candidate rate を追加する。
- これは train-side oracle readout であり、direct prediction score ではないことを summary notes に明記する。

exp202 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-06 follow-up 9 design

exp202 v2 の path artifact を使い、TVT panel に local path segments を重ねる。v13 の `pred_top1_tvt` center-line は廃止し、NPZ に保存された `pred_tvt_path` を正とする。

入力:

- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`

読み込み:

- NPZ から `pred_tvt_path`, `md_path`, `sample_id`, `horizontal_offsets` を読む。
- sample CSV から `path_npz_sample_index`, `sample_id`, `well`, `row_center`, `md_since_prefix` を読む。
- rank CSV は schema / rank metrics 記録用に読み、summary と source SHA に残す。
- `sample_id` の NPZ order と sample CSV order が一致することを検証する。

描画:

- well ごとに sample CSV を filter する。
- 各 sample について、`horizontal_offsets == 0` の step を center とし、`x_path = md_path - md_path[center] + md_since_prefix` で plot 座標へ変換する。
- `x_path` は該当 well の plot 対象 `md_since` 範囲に clip する。
- rank1 path は濃い紫、rank2-10 path は薄い紫の `LineCollection` として描く。
- center 候補 marker は小さく薄く残し、path segment の中心位置確認に使う。

注意:

- exp202 path は local 128-row window segment で、full-well trajectory に stitch したものではない。
- train-side visualization 専用で、hidden true TVT を guide 生成や選択には使わない。
- exp202 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-06 follow-up 10 design

v14 の local segment 表示は、exp202 の保存物を正しく読んでいるが、短い window segment 群として見えるため path 可視化として期待とずれる。plot 側で saved local windows を rank1 stitched guide に集約する。

入力は follow-up 9 と同じ。

手順:

- well ごとに `candidate_path_samples` を `md_since_prefix` 昇順に並べる。
- 各 sample の `path_npz_sample_index` で NPZ の `pred_tvt_path` と `horizontal_row_index` を読む。
- x 座標は `horizontal_row_index[sample] - prefix_end` とする。これにより raw horizontal row index を plot の `md_since` 系へ戻す。
- x が該当 well の plot 範囲外、または TVT が非 finite の点は除外する。
- rank1 (`EXP202_STITCH_RANK=1`) の `pred_tvt_path` だけを使い、同じ integer x に複数 window から予測が来た場合は median を取る。
- 集約後の `(x, median_tvt)` を x 昇順で1本の紫線として描画する。

注意:

- これは visualization aggregation で、exp202 が native に出力した full-well trajectory ではない。
- rank2-10 は今回の plot では描かない。top10 の価値は従来通り candidate-union oracle metrics として title / summary に残す。
- exp202 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-06 follow-up 11 design

v16 の full-step stitched guide は、exp202 path artifact を正しく読んでいるが、model の full local path logits が center 以外で大きく飛ぶため TVT panel の y-scale を壊す。画像確認では `00e12e8b` の purple line が true TVT 付近 11600 ft に対して 11341-11774 ft まで広がっていた。

修正方針:

- `pred_tvt_path[:, rank, step]` 全 step の stitching は使わない。
- scale が合っている center prediction だけを使う。
- `candidate_path_rank_index.csv.gz` から `rank == 1` の `center_pred_tvt` を読む。
- `candidate_path_samples.csv.gz` から `path_npz_sample_index`, `md_since_prefix`, `prefix_end` を読み、rank index と join する。
- well ごとに `md_since_prefix` 昇順で `center_pred_tvt` を接続する。
- line は `exp202 rank1 center path` と明記し、full-well trajectory ではなく sparse center candidate path であることを summary に残す。
- manifest に `exp202_path_rank1_center_points`, `exp202_path_rank1_center_min`, `exp202_path_rank1_center_max` を追加する。
- TVT panel の y-axis は true TVT、PF/Beam、exp148 ML OOF、`-Z likPF minmax` の finite range から margin 付きで固定し、exp202 center candidates はこの trusted range 内だけ描画する。

exp202 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-05 follow-up 8 design

exp202 overlay の top1 表示を、点から top1 center path line に変更する。

入力は follow-up 7 と同じ `heatmap_candidates` を使う。exp202 train 実装は内部で window 内 `pred_path_tvt` を計算しているが、保存済み CSV には full path 配列を保存していない。そのため v12 plot 側では、保存済み `pred_top1_tvt` の sample center を well ごとに `md_since` 昇順で接続する。

描画:

- `heatmap_candidates` を `well`, `md_since` で sort する。
- finite な `md_since` と `pred_top1_tvt` を `ax.plot` で接続し、紫の線と小 marker で表示する。
- `pred_top2_tvt` から `pred_top10_tvt` は従来通り薄い紫 point として表示する。
- label / summary は `exp202 heatmap top1 path` とし、strict full path ではなく top1 center path であることを notes に残す。

exp202 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-06 follow-up 12 design

exp207 の stitched path rows を、v12 plot notebook の heatmap-MDN overlay として描画する。exp202 の sparse center candidate ではなく、exp207 が target-free stitch した row-level path candidate を使う。

入力:

- `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_stitched_path_rows.csv.gz`
- `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_by_well.csv`
- `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_candidate_union_metrics.csv`
- `exp207_heatmap_mdn_overlapping_window_path_stitch_probe_source_window_coverage.csv`

手順:

- exp207 stitched rows を読み、`id/well` で exp072 feature-cache plot frame の `md_since` を付与する。
- well ごとの repeated scan を避けるため、`plot_frame` と exp207 stitched rows は groupby index を事前に作る。
- `stitched_candidate_rank` 1-3 を rank 別に描く。
- rank1 は濃い紫線 + 小 marker、rank2-3 は薄い紫線にする。
- x 範囲は対象 well の plot frame 範囲、y 範囲は true/PF/Beam/ML/-Z guide から作った trusted axis range に制限する。
- by-well title は exp207 `union_oracle_rmse`, `existing_oracle_rmse`, `new_best_candidate_rate` を表示する。

注意:

- exp207 は covered-row candidate-union oracle diagnostic であり、direct prediction score ではない。
- exp207 の parent result では stitched-only top3 が弱いことが確認済みなので、plot は診断専用とする。
- exp207 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 13 design

exp207 stitched path overlay を exp210 full-well candidate path overlay に差し替える。exp210 は full-well path contract として `md_from_ps`, `path_rank`, `tvt_pred` を保存しているため、plot 側では exp072 feature-cache へ `id/well` join して x 座標を補う処理を廃止し、artifact の座標を直接使う。

入力:

- `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz`
- `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_candidate_union_by_well.csv`
- `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_candidate_union_metrics.csv`
- `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_source_window_coverage.csv`
- `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_contract_metrics.csv`

手順:

- full-well candidate paths は描画に必要な `well`, `md_from_ps`, `path_rank`, `tvt_pred` だけを読む。
- well ごとの repeated scan を避けるため、`exp210_full_well_candidate_paths.groupby("well").indices` を作る。
- `path_rank` 1-5 を描き、rank1 は濃い紫線 + 小 marker、rank2-5 は薄い紫線にする。
- x 範囲は対象 well の exp072 plot frame 範囲、y 範囲は true/PF/Beam/ML/-Z guide から作った trusted axis range に制限する。exp210 out-of-range path point は autoscale に使わない。
- by-well title は exp210 `union_oracle_rmse`, `existing_oracle_rmse`, `new_best_candidate_rate` を表示する。
- summary には full-well candidate path SHA、contract metrics、candidate union metrics を残す。

注意:

- exp210 は covered-row candidate-union oracle diagnostic であり、direct prediction score ではない。
- exp210 parent result では full-well candidate 単独 path は弱く、価値は PF/Beam candidate union への oracle headroom として扱う。
- exp210 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 14 design

exp210 full-well candidate path overlay を exp212 full-grid candidate path overlay に差し替える。exp212 は full-grid path contract として `md_since`, `md_from_ps`, `path_rank`, `tvt_pred`, `coverage_flag`, `fallback_flag` を保存しているため、plot 側では `md_since` を TVT panel の x 座標として使う。

入力:

- `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz`
- `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_candidate_union_by_well.csv`
- `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_candidate_union_metrics.csv`
- `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_source_window_coverage.csv`
- `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_contract_metrics.csv`

手順:

- full-grid candidate paths は描画に必要な `well`, `md_since`, `md_from_ps`, `path_rank`, `tvt_pred`, `coverage_flag`, `fallback_flag` だけを読む。
- well ごとの repeated scan を避けるため、`exp212_full_grid_candidate_paths.groupby("well").indices` を作る。
- `path_rank` 1-5 を描き、rank1 は濃い紫線 + 小 marker、rank2-5 は薄い紫線にする。
- x 軸は exp209 enriched cache の `id/well` を exp072 plot frame に join して取得した `md_since` を使う。
- x 範囲は対象 well の exp072 plot frame 範囲に制限する。
- y 範囲では exp212 rank1 を事前に落とさず、true/PF/Beam/ML/-Z guide から作った trusted axis range に rank1 min/max を追加して表示範囲を決める。
- by-well title は exp212 `union_oracle_rmse`, `existing_oracle_rmse`, `new_best_candidate_rate` を表示する。
- manifest には exp212 rank1 の plot point 数、raw point 数、source/fallback point 数、fallback rate、coverage rate、trusted/display y range を残す。
- summary には full-grid candidate path SHA、contract metrics、candidate union metrics を残す。

注意:

- exp212 は covered-row candidate-union oracle diagnostic であり、direct prediction score ではない。
- exp212 parent result では stitched-only top5 が弱く、価値は既存 candidate union への oracle headroom として扱う。
- exp212 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 15 design

exp212 full-grid candidate path overlay を exp215 learned MTP full-tail path overlay に差し替える。exp215 は full-grid path rows として `md_since`, `row_index`, `path_rank`, `tvt_pred`, `path_prob`, `weighted_tvt_pred`, `source_window_count`, `coverage_flag`, `fallback_flag`, `candidate_cost` を保存しているため、plot 側では `md_since` を TVT panel の x 座標として使う。

入力:

- `exp215_mtp_full_tail_heatmap_path_generator_probe_full_grid_candidate_paths.csv.gz`
- `exp215_mtp_full_tail_heatmap_path_generator_probe_candidate_union_by_well.csv`
- `exp215_mtp_full_tail_heatmap_path_generator_probe_candidate_union_metrics.csv`
- `exp215_mtp_full_tail_heatmap_path_generator_probe_summary.json`

手順:

- full-grid candidate paths は描画と manifest に必要な列だけを読む。
- well ごとの repeated scan を避けるため、`exp215_full_grid_candidate_paths.groupby("well").indices` を作る。
- `path_rank` 1-5 を描き、rank1 は濃い紫線 + 小 marker、rank2-5 は薄い紫線にする。
- `weighted_tvt_pred` は rank1 rows から x ごとに median 集約し、オレンジ破線で描く。
- x 範囲は対象 well の exp072 plot frame 範囲に制限する。
- y 範囲では exp215 rank1 と weighted path を事前に落とさず、true/PF/Beam/ML/-Z guide から作った trusted axis range に rank1 min/max と weighted min/max を追加して表示範囲を決める。
- by-well title は exp215 `existing_plus_learned_oracle_rmse`, `existing_oracle_rmse`, `fallback_row_rate` を表示する。
- manifest には exp215 rank1 の plot point 数、raw point 数、source/fallback point 数、fallback rate、coverage rate、path probability / cost / source window count 平均、weighted path の点数と TVT 範囲を残す。
- summary には exp215 full-grid candidate path SHA、summary JSON、candidate union metrics を残す。

注意:

- exp215 の `candidate_score` / `candidate_cost` は path decoder 側の heuristic score に近く、hengck23 notebook の learned mode probability と同一視しない。
- exp215 parent result では learned MTP weighted 単体 path は弱く、価値は既存 candidate union への oracle headroom として扱う。
- exp215 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 16 design

exp215 learned MTP full-tail path overlay を外し、exp209 HMM direct-comparison output overlay に差し替える。exp209 は candidate path generator ではなく row-level HMM feature / blend readout なので、path rank ではなく `hmm_mean_tvt` と best blend `blend_likpf_hmm_w500` を線として描く。

入力:

- `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz`
- `exp209_vs_exp072_exp205_by_well_delta.csv`
- `exp209_vs_exp072_exp205_overall_metrics.csv`
- `exp209_vs_exp072_exp205_summary.json`

手順:

- enriched HMM cache は `id`, `well`, `md_since`, `hmm_mean_tvt`, `blend_likpf_hmm_w500`, `hmm_std`, `hmm_loglik`, `hmm_minus_likpf_mean` だけを読む。
- well ごとの repeated scan を避けるため、`exp209_enriched_hmm.groupby("well").indices` を作る。
- `hmm_mean_tvt` は紫線、`blend_likpf_hmm_w500` はオレンジ破線で描く。
- x 範囲は対象 well の exp072 plot frame 範囲に制限する。
- y 範囲では HMM mean と best blend を事前に落とさず、true/PF/Beam/ML/-Z guide から作った trusted axis range に HMM / blend min/max を追加して表示範囲を決める。
- by-well title は exp209 `hmm_mean_tvt`, `blend_likpf_hmm_w500`, `exp072_likpf_mean` の RMSE を表示する。
- manifest には exp209 HMM rows、HMM / blend point 数、TVT min/max、HMM std/loglik/likPF 差分平均、by-well RMSE を残す。
- summary には exp209 enriched cache SHA、by-well delta、overall metrics、summary JSON を残す。

注意:

- exp209 は full path candidate output ではなく row-level HMM mean / blend feature output である。
- `blend_likpf_hmm_w500` は exp209 の train-side best comparison candidate で、hidden-test prediction score ではない。
- exp209 は保存済み output を読むだけで、新規学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 17 design

follow-up 16 の HMM overlay を簡素化し、背景地層の scale 表示を修正する。前回版では地層境界 `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` も背景カーブの共通 min-max rescale 対象に入っていたため、公式画像や discussion 図と比べると地層の縦位置がずれて見える可能性がある。地層列は TVT/depth と同じ座標系の境界値として扱い、plot y-range への再スケーリングを行わない。

手順:

- `BACKGROUND_COLUMNS` の formation columns は `already_tvt_scale=True`, `use_common_scale=False` とし、raw `raw_ANCC` などをそのまま描く。
- `BACKGROUND_BANDS` の fill_between も raw formation boundary 値の間を塗る。
- 地層背景は TVT panel の axis range driver には使わず、true/PF/Beam/ML/-Z guide と exp209 HMM mean で表示 y range を決める。
- exp209 enriched HMM cache から読む overlay 列は `hmm_mean_tvt` のみにする。`hmm_std`, `hmm_loglik`, `hmm_minus_likpf_mean` は manifest 診断用に保持する。
- `add_exp209_hmm_outputs()` は `hmm_mean_tvt` だけを描画し、y range に含める。
- title / manifest / summary から `blend_likpf_hmm_w500` 関連の point 数、TVT range、RMSE を削除する。
- summary の exp209 overall metrics は `exp072_likpf_mean` と `hmm_mean_tvt` に絞る。

注意:

- 地層列が raw 座標で画面外に出る場合でも、背景だけで y-axis を広げない。予測値比較の視認性を優先する。
- 今回は可視化 notebook のみを更新する。モデル学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 18 design

v22 では formation 列を raw 値のまま TVT 軸に描いたため、`TVT` が 11000-13000 付近である一方、formation 列は `Z` と同じ -9000 付近の座標になり、完全に画面外へ出た。formation 列は TVT そのものではなく、horizontal well の `Z` 座標系上の地層境界値として扱う。

変換:

- 各 row について `formation_tvt = raw_TVT + raw_Z - formation_Z` を計算する。
- `formation_Z` が current well `raw_Z` より小さい、つまりより深い場合、`raw_Z - formation_Z` が正になり、TVT は大きくなる。
- この変換後の formation TVT を line / band の y 座標として使う。

y-axis:

- まず true/PF/Beam/ML/-Z guide と exp209 HMM mean から通常の表示範囲を作る。
- 変換済み formation TVT の median を boundary 値として集める。
- 通常表示範囲を挟む最も近い上側/下側 boundary と、範囲内 boundary だけを display y range に追加する。
- これにより、対象 TVT 周辺の地層 band は見えるが、全 stratigraphy を入れて予測線を必要以上に圧縮しない。

manifest / summary:

- `formation_axis_context_points`, `formation_axis_context_min`, `formation_axis_context_max` を manifest に出す。
- summary の formation method は `formation_z_to_tvt_delta` とし、formula を明記する。

注意:

- common min-max rescale は使わない。
- `likPF/HMM blend` は引き続き描画しない。
- 今回は可視化 notebook のみを更新する。モデル学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 19 design

地層表示自体を診断 plot から外す。TVT panel は true TVT、exp148 ML OOF、PF/Beam 系、`-Z likPF minmax` guide、exp209 HMM mean の比較に絞る。

手順:

- `BACKGROUND_COLUMNS` は `z_likpf_minmax` のみにする。
- `BACKGROUND_BANDS` は空配列にする。
- formation Z-to-TVT 変換、formation context y-axis 追加、formation context manifest 列を削除する。
- summary の `visual_guides.formation_background` は `plotted: false` として残す。
- plot file path は `PLOTS_DIR / f"{well_id}.png"` とし、`all_wells__` prefix を使わない。
- zip 内の arcname も PNG basename のままなので、zip 内も `{well}.png` になる。

注意:

- `plot_scope` は引き続き all wells を意味するため summary 上は `all_wells` のままでもよい。削除対象は PNG file name の `all_wells__` prefix。
- `-Z likPF minmax` guide は地層ではないため残す。
- 今回は可視化 notebook のみを更新する。モデル学習、PF/Beam 再生成、hidden inference、submission は行わない。

## 2026-07-07 follow-up 20 design

exp209 HMM mean の uncertainty を同じ TVT panel に薄い band として重ねる。入力は既存の `exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz` に含まれる `hmm_mean_tvt` と `hmm_std` を使う。

手順:

- `add_exp209_hmm_outputs()` で `hmm_mean_tvt` の有効点を作ったあと、`hmm_std` が finite かつ 0 以上の点を band 用に抽出する。
- `lower = hmm_mean_tvt - 2*hmm_std`、`upper = hmm_mean_tvt + 2*hmm_std` を計算する。
- 同一 `md_since` が複数ある場合は `lower` / `upper` を median 集約し、`md_since` 昇順に並べる。
- `ax.fill_between()` で紫系の低 alpha band を先に描き、その上に従来の HMM mean 線を描く。
- `hmm_2sigma_min` / `hmm_2sigma_max` を `exp209_overlay_range()` に追加し、band が y-axis 外に切れないようにする。
- manifest に `exp209_hmm_2sigma_segments`, `exp209_hmm_2sigma_points`, `exp209_hmm_2sigma_min`, `exp209_hmm_2sigma_max` を出す。
- summary の `exp209_overlay` に `hmm_std_column`, band style, formula を出す。

注意:

- `hmm_std` は exp209 HMM output の posterior spread 由来であり、可視化上の guide として扱う。厳密に calibrated された 95% interval とは仮定しない。
- 地層 line / band は描画しない。
- `likPF/HMM blend` は引き続き描画しない。
- 今回は可視化 notebook のみを更新する。モデル学習、PF/Beam 再生成、hidden inference、submission は行わない。
