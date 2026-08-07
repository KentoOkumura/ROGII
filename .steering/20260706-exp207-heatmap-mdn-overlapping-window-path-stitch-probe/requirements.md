# 要件

## 依頼

`heatmap_mdn_overlapping_window_path_stitch_probe` backlog を `exp207_heatmap_mdn_overlapping_window_path_stitch_probe` として実装する。discussion 699853 の multi-trajectory / top paths 方向に寄せ、exp202 の local window path を full-well path candidate として stitch できるか train-side で診断する。

## 制約

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- 比較対象: `exp099_pf_multi_observation_likelihood_probe` の PF/Beam candidate cache
- 初回は selector 学習、推論、提出をしない。
- exp202 v2 の local 128-row path artifact は sparse validation sample 出力であり、full-well dense trajectory ではない。coverage / overlap 不足を検出し、結果解釈に残す。
- stitch score に true TVT、oracle best、abs-error、within10、target-in-grid、true-error rank を使わない。
- target は stitched path が固定された後の train-side oracle readout にだけ使う。
- direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend はしない。
- 再現性: `docs/06_reproducibility.md` に従い、入力 artifact SHA、gzip decompressed SHA、runtime、upstream stochastic component を記録する。

## 受け入れ基準

- `.steering/20260706-exp207-heatmap-mdn-overlapping-window-path-stitch-probe/` に requirements / design / tasklist がある。
- `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/` に config、helper、train / inference notebook source、notebook、README、SESSION_NOTES、result、metrics placeholder がある。
- train notebook が親実験、route、stitch topK / beam / output topN、入力 artifact、既存 candidate set、出力生成物を表示する。
- helper が exp202 path artifact と exp099 candidate cache を読み、target-free score で topN stitched paths を作る。
- helper が row-level stitched path、window assignment、source coverage、candidate union metrics、distance bucket、by-well readout、summary JSON を保存する。
- static validation と Jupytext conversion が通る。
- Kaggle push 前のコストガードとして、GPU なし、LightGBM 0 configs / 0 boosters、parent/control retraining なしを `SESSION_NOTES.md` に記録する。
