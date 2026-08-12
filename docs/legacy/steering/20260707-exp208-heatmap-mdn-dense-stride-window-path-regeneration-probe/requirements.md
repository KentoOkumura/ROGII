# 要件

## 依頼

`heatmap_mdn_dense_stride_window_path_regeneration_probe` backlog を実験化する。exp207 は exp202 の sparse local path artifact を stitch しただけで、source overlap が 773 wells 中 3 wells / 39 pairs に留まった。exp208 では exp202 の saved model artifact から validation wells の dense stride local paths を再生成し、overlap 付き stitch readout をやり直す。

## 制約

- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- stitch 比較元: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- 比較 candidate cache: `exp099_pf_multi_observation_likelihood_probe`
- exp202 parent/control の再学習はしない。saved fold model を読むだけにする。
- LightGBM config 数 0、booster 数 0、submit なし。
- 初回は stride 64、必要なら後続で stride 32 を検討する。
- local topK は 5 / 10 を比較する。
- stitch score に `true_tvt_path`、`true_center_tvt`、`center_abs_error`、oracle best、abs-error、within10 を入れない。
- positive でも direct replacement、softmax average、PF weight replacement、inference、submit には進めない。
- 再現性: `docs/06_reproducibility.md` に従い、saved model、input cache、生成物の SHA を記録する。

## 受け入れ基準

- `.steering`、`experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/`、`config.yaml`、train / inference notebook source、helper、記録ファイルが揃っている。
- train notebook で、exp202 saved model、raw train data、exp099 candidate cache の入力確認、dense sample 数、stride、fold、topK、保存物を確認できる。
- dense path artifact を生成し、local topK 5 / 10 の stitch readout を保存できる。
- 評価は source overlap pair count、row coverage、gap boundary abs、overlap disagreement、stitched only oracle、existing + stitched oracle、`1000_plus`、worst-well、path step / curvature を含む。
- deterministic anchor として扱わず、exp202 GPU-trained artifact 依存であることを記録している。
- gzip 生成物は decompressed content SHA を主証拠として記録している。
- `py_compile`、`ruff --select F821,E501`、Jupytext conversion / `--test`、`make validate-exp` が通る。
