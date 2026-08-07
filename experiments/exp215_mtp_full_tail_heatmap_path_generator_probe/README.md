# exp215_mtp_full_tail_heatmap_path_generator_probe

## 状態

Kaggle train v1 完了。train-side diagnostic のため inference / submit は行わない。

## 仮説

exp212 の full-grid heatmap path artifact は source coverage が 43.0% に留まり、残り 57.0% が endpoint hold / interpolation fallback になった。heuristic stitch ではなく learned `path_logit` を持つ MTP full-tail generator を学習すれば、rank1 / weighted path / topK candidate が full-grid でより plausible になり、既存 PF/Beam candidate union に追加したときの oracle headroom も維持できる可能性がある。

## 検証方針

- exp202 と同じ 5ch heatmap input を使う。
- `path_pred [K,L]` と `path_logit [K]` を出す continuous MTP head を学習する。
- closest-mode path regression + mode CE loss を使う。
- valid fold の dense full-tail windows から full-grid candidate path artifact を生成する。
- exp099 candidate cache と join し、existing union、learned MTP topK、weighted path、existing+learned topK の oracle RMSE / within10 / by-well / distance bucket を比較する。
- `TVT_input` history SDF は入力 channel として使うが、SDF output head、SDF target、`sdf_loss` は使わない。
- inference / submit はしない。

## 所見

full-grid contract は coverage 1.0 / fallback unique row rate 0.0 で成立し、exp212 の fallback-heavy 問題は解消した。existing + learned MTP top5 oracle RMSE は 7.434030 -> 5.113655 と改善。一方、learned MTP weighted path は RMSE 59.272142 と弱いため、direct replacement / weighted submit には使わない。

## 実行

Kaggle GPU train は 1 active spec、5 folds、5 CNN models、LightGBM 0 configs / 0 boosters、parent/control retraining なしで完了した。
