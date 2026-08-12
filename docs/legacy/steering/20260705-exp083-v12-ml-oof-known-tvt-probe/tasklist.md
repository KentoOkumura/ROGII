# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 再現性設計を `design.md` に記入した。
- stochastic 処理なし、PF/Beam 再生成なし、GPU 学習なしとして整理した。
- 別名 Jupytext percent notebook を実装した。
- `.ipynb` へ変換し、構文チェック、`jupytext --test`、`ruff --select F821` を実行した。
- `SESSION_NOTES.md` に作成ファイル、入力、検証結果、Kaggle 実行時に必要な input source を記録した。
- known tail fit 版 v6 は output を取得せず、status/logs のみ確認した。
- `-Z` guide を known anchor + typewell tail hybrid に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle に v7 として push した。
- ユーザー指示により v7 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v7 完了後に status/logs のみ確認し、output は取得しなかった。
- v7 は typewell TVT 範囲外へ外挿される実装で、意図と違うことを確認した。
- `-Z` guide を known-tail direction-aware typewell-tail min-max scaling に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle に v8 として push して実行した。
- v8 完了後に status/logs のみ確認し、output は取得しなかった。
- follow-up 4 の要件と設計を追記した。
- typewell anchored guide と `-Z` likPF minmax guide を実装した。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle に v9 として push した。
- ユーザー指示により v9 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- follow-up 5 の要件と設計を追記した。
- `-Z` likPF minmax を last known TVT anchor 版に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle に v10 として push した。
- ユーザー指示により v10 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v10 完了後に status/logs のみ確認し、output は取得しなかった。
- v10 の `-Z likPF minmax anchored` は clip 有効で、上下の張り付きは clip による可能性が高いことを確認した。
- follow-up 6 の要件と設計を追記した。
- `-Z` guide を direct `-Z` to `likpf_mean` min-max 版に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle に v11 として push した。
- ユーザー指示により v11 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v11 完了後に status/logs のみ確認し、output は取得しなかった。
- follow-up 7 の要件と設計を追記した。
- exp202 heatmap-MDN candidate overlay を v12 plot notebook に追加した。
- Jupytext notebook と Kaggle package を再生成した。
- Kaggle package metadata に exp202 kernel source を追加した。
- Kaggle に v12 として pushした。
- ユーザー指示により v12 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v12 完了後に status/logs のみ確認し、output は取得しなかった。
- exp202 の保存済み `validation_predictions` / `heatmap_candidates` は sample center の topK TVT 候補と path step サマリを保存しているが、モデル内部で生成した window 内 `pred_path_tvt` 配列は保存していないことを確認した。
- follow-up 8 の要件と設計を追記した。
- exp202 overlay の `pred_top1_tvt` を `md_since` 昇順で接続する top1 center path line に変更した。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、notebook JSON check が通った。
- Kaggle に v13 として push した。
- push 後に同じ kernel id の metadata pull と status 確認を行い、RUNNING を確認した。
- ユーザー指示により v13 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v13 完了後に status/logs のみ確認し、output は取得しなかった。
- exp202 `pred_top1_tvt` は意図通り参照しているが、これは window 内 full path ではなく sample center の top1 候補であり、top1 単体の RMSE は約 60.49 と悪いことを確認した。
- follow-up 9 の要件と設計を追記した。
- exp202 v2 の saved local path artifact (`heatmap_candidate_paths_top10.npz` + sample/rank index CSV) を読むように plot notebook を変更した。
- v13 の `pred_top1_tvt` center-line ではなく、NPZ の `pred_tvt_path` を rank1 濃線、rank2-10 薄線として描くようにした。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、package notebook JSON check が通った。
- Kaggle に v14 として push した。
- push 後に同じ kernel id の metadata pull と status 確認を行い、RUNNING を確認した。
- ユーザー指示により v14 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v14 完了後の logs から、exp202 path artifact は正しく読めているが、plot は local 128-row segment 群であり期待する path 表示と違うことを確認した。
- follow-up 10 の要件と設計を追記した。
- exp202 saved local path windows の rank1 を `horizontal_row_index - prefix_end` で global x に戻し、x ごとに median 集約する stitched guide 表示へ変更した。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、package notebook JSON check が通った。
- local exp202 artifact smoke で rank1 stitched guide が生成できることを確認した。
- Kaggle に v15 として push した。
- v15 は `read_exp202_candidate_path_samples()` で `prefix_end` を保持しておらず、描画時に `AttributeError: 'Pandas' object has no attribute 'prefix_end'` で失敗した。
- `prefix_end` を読み込み列に追加し、欠損時は `row_center - md_since_prefix` で復元する fallback を入れた。
- v16 修正版として Jupytext notebook と Kaggle package を再生成した。
- v16 修正版の構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、package notebook JSON check が通った。
- Kaggle に v16 として push した。
- push 後に同じ kernel id の metadata pull と status 確認を行い、RUNNING を確認した。
- 10 分程度の logs 監視と通常 logs 取得ではログ本文なし。status は RUNNING のまま。
- ユーザー指示により v16 の logs 監視だけ停止した。Kaggle 実行自体は継続。
- v16 完了後に status/logs を確認し、COMPLETE を確認した。
- v16 の代表画像 `00e12e8b` を 1 枚だけ取得し、purple `exp202 stitched path rank1` だけが y-scale を壊していることを確認した。
- local exp202 artifact 再計算で、full `pred_tvt_path` stitching は center 以外の step が大きく飛ぶことを確認した。
- follow-up 11 の要件と設計を追記した。
- full-step stitching を廃止し、`candidate_path_rank_index.csv.gz` の rank1 `center_pred_tvt` を `md_since_prefix` 順につなぐ center path 表示へ変更した。
- true/PF/Beam/ML/-Z guide で TVT panel の y-axis を固定し、exp202 out-of-range center candidate が autoscale を壊さないようにした。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、package notebook JSON check が通った。
- Kaggle に v17 として push した。
- v17 の完了後 status/logs を確認し、COMPLETE を確認した。
- 代表画像 `00e12e8b` を 1 枚だけ取得し、purple exp202 line による y-axis scale 破壊が解消されていることを確認した。
- follow-up 12 の要件と設計を追記した。
- exp202 rank1 center path overlay を exp207 `stitched_path_rows.csv.gz` の rank1-3 stitched path overlay に差し替えた。
- output prefix を `...exp207_stitchedpath_all` に変更した。
- Kaggle package metadata の kernel source を exp202 から `kentookumura/exp207-hmdn-path-stitch-train` に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle push / run はまだ実行していない。
- follow-up 13 の要件と設計を追記した。
- exp207 stitched path overlay を exp210 full-well candidate path overlay に差し替えた。
- output prefix を `...exp210_fullwellpath_all` に変更した。
- Kaggle package metadata の kernel source を `kentookumura/exp210-hmdn-full-well-path-generation-train` に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- follow-up 14 の要件と設計を追記した。
- exp210 full-well path overlay を exp212 full-grid candidate path overlay に差し替えた。
- output prefix を `...exp212_fullgridpath_all` に変更した。
- exp212 path の x 軸を `md_since` にし、rank1 を y 範囲で落とさず display y range に含めるようにした。
- manifest に exp212 rank1 fallback/source point 数、fallback rate、coverage rate、display y range を追加した。
- Kaggle package metadata の kernel source を `kentookumura/exp212-hmdn-full-grid-path-generation-train` に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- follow-up 15 の要件と設計を追記した。
- exp212 full-grid path overlay を exp215 learned MTP full-tail full-grid candidate path overlay に差し替えた。
- output prefix を `...exp215_fulltailmtp_all` に変更した。
- exp215 の rank1-5 `tvt_pred` と rank1 rows の `weighted_tvt_pred` を TVT panel に描画するようにした。
- manifest / summary / title 指標を exp215 `existing_plus_learned_mtp_topk` と `existing_union` の candidate-union readout に差し替えた。
- Kaggle package metadata の kernel source を `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train` に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle push / run はこれから実行する。
- follow-up 16 の要件と設計を追記した。
- exp215 learned MTP full-tail overlay を描画対象から外した。
- exp209 enriched HMM output を読み、`hmm_mean_tvt` と `blend_likpf_hmm_w500` を TVT panel に描画するようにした。
- title / manifest / summary 指標を exp209 by-well HMM / best blend / likPF RMSE に差し替えた。
- Kaggle package metadata の kernel source を `kentookumura/exp209-joint-exact-parity-train` に差し替えた。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle に v21 として push した。
- v21 完了後に status/logs のみ確認し、output は取得しなかった。
- v21 は 773 well の exp209 HMM mean / best blend overlay plot、manifest、plots zip、summary を生成完了した。

## 2026-07-07 follow-up 17

- ユーザー確認により、背景に薄く表示している地層のスケールがずれて見えること、`likPF/HMM blend` は不要であることを確認した。
- 地層背景 `ANCC` / `ASTNU` / `ASTNL` / `EGFDU` / `EGFDL` / `BUDA` を raw TVT/depth scale のまま描画するように変更した。
- 地層 label から `scaled` 表記を削除した。
- exp209 overlay を `hmm_mean_tvt` のみへ変更し、`blend_likpf_hmm_w500` は読み込み・描画・title・manifest・summary から削除した。
- output prefix を `...exp209_hmm_rawformation_all` に変更した。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle に v22 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。完了判定は後続の status/logs で確認する。

## 2026-07-07 follow-up 18

- ユーザー確認により、v22 では地層背景が全く表示されていないことを確認した。
- v22 logs から 773 well の plot 生成は COMPLETE しており、`...exp209_hmm_rawformation_all` の manifest / plots zip / summary が生成済みであることを確認した。
- 原因は formation 列が `TVT` ではなく `Z` と同じ -9000 付近の座標系であり、raw 値のまま TVT 軸へ描いたため画面外に出たこと。
- 地層列の変換を `formation_tvt = raw_TVT + raw_Z - formation_Z` に変更した。
- TVT y-axis に、通常の予測範囲を挟む近傍 formation bracket を追加するようにした。
- manifest に `formation_axis_context_points`, `formation_axis_context_min`, `formation_axis_context_max` を追加した。
- output prefix を `...exp209_hmm_formationztvt_all` に変更した。
- `likPF/HMM blend` は引き続き削除済み。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle に v23 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。完了判定は後続の status/logs で確認する。

## 2026-07-07 follow-up 19

- ユーザー確認により、地層 plot は不要、PNG 名の `all_wells` prefix も不要であることを確認した。
- v23 は status/logs で COMPLETE を確認した。
- v23 は 773 well の formation Z-to-TVT 背景 plot、manifest、plots zip、summary を生成完了したが、output は取得していない。
- 地層境界 line と地層 filled band を完全に描画対象から外した。
- formation context の y-axis 追加と manifest 列を削除した。
- `-Z likPF minmax` guide は残した。
- PNG 保存名を `all_wells__{well}.png` から `{well}.png` に変更した。
- output prefix を `...exp209_hmm_noformation_all` に変更した。
- `likPF/HMM blend` は引き続き削除済み。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle に v24 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。完了判定は後続の status/logs で確認する。

## 2026-07-07 follow-up 20

- ユーザー指示により、HMM の 2sigma band を薄い色で TVT panel に追加する。
- v24 は COMPLETE 済みで、773 well の no-formation plot を生成済み。output は取得しない。
- `hmm_mean_tvt +/- 2*hmm_std` を `fill_between` で描画するようにした。
- band は `md_since` ごとに median 集約し、HMM mean 線の下に薄い紫色で描画する。
- band の min/max を y-axis range に含めるようにした。
- manifest / summary に `exp209_hmm_2sigma_*` と band formula を追加した。
- 地層 plot、PNG の `all_wells__` prefix、`likPF/HMM blend` は引き続き削除済み。
- output prefix を `...exp209_hmm_2sigma_noformation_all` に変更した。
- Jupytext notebook と Kaggle package を再生成した。
- 構文チェック、`ruff --select F821`、`jupytext --to ipynb --test`、local/package notebook JSON check が通った。
- Kaggle に v25 として push した。
- push 後 status は `KernelWorkerStatus.RUNNING`。実行中 logs は空。
- ユーザー指示により監視だけ停止した。Kaggle 側実行は継続中。
- ユーザー完了連絡後に status/logs を確認し、v25 は `KernelWorkerStatus.COMPLETE`。
- output prefix は `...exp209_hmm_2sigma_noformation_all`。
- v25 は 773 well の plot、manifest、plots zip、summary を生成完了した。
- manifest rows は 773、`Z-to-likPF simple minmax status` は `{'ok': 773}`、coverage min は 1.0。
- exp209 HMM mean points per well min/max は 407 / 10052。
- exp209 HMM +/-2sigma points per well min/max は 407 / 10052。
- exp209 HMM +/-2sigma TVT range は 10027.519931 / 12892.8001084。
- output は取得していない。
