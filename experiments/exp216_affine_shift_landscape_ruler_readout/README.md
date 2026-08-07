# exp216_affine_shift_landscape_ruler_readout

## 状態

- ルート: `pf_beam`
- 状態: completed_train_side_rejected_no_submit
- CV: diagnostic_only
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-07
- 親/参照: `KAGGLE_DIRECTION.md` backlog `affine_shift_landscape_ruler_readout`、`exp167_fft_denoised_gr_matching_audit`、`exp072_exp063_full_replay_feature_cache`

## 仮説

known prefix の `TVT_input` を使った affine 形式の GR calibration が、hidden-tail と prefix-backtest の shift-landscape における localization 曖昧性を減らすかを監査する。
`raw`、`rolling_median_11`、`savgol_31_p2` の filter それぞれで校正あり/なしを比較し、row-level readout で fallback 判定が作れることを狙う。

## 変更点

- 既存の affine shift-scan flow を継承。
- 行レベルで以下の ruler 指標を追加保存。
  - `best_shift_ft` / `best_delta`
  - `second_shift_ft` / `second_delta`
  - `second_delta_vs_best`
  - `second_cost`
  - `margin`（= `top1_top2_cost_gap`）
  - `zero_shift_ft`
  - `zero_cost`
  - `zero_rank`
  - `secondary_mode_shift_ft` / `secondary_mode_gap`
  - `bimodal_flag`
  - `prefix_holdout_error` / `prefix_holdout_abs_error`
  - `calibration_residual_scale`（`fit_calibration` の residual MAD）
- 集約した shift curve と、distance bucket 別の誤差相関を生成物として保存する。
- fixed exp072 cache candidates の observation cost は readout 用に保持し、`inference`・`submission` には使用しない。
- ML 学習、candidate replacement、inference port、submission は行わない。

## 検証方針

- 計測井の `prefix_backtest` + `hidden_tail` 行を対象。
- primary: `rmse`
- secondary: `mae`、`within2`、`within5`、`within10`
- readout 指標: `margin`、`entropy`、`decoy_gap_15_25ft`、`zero_rank`、`bimodal_flag`
- route は diagnosis のみ。悪化系/不安定なら採用しない。

## 所見

- Kaggle train v1 は 773 wells / 3,561,984 row-context rows で完走。
- best overall は `savgol_31_p2__raw` で RMSE 108.534313、MAE 69.576973。
- hidden_tail best は `rolling_median_11__raw` で RMSE 125.707127、MAE 76.419067。
- affine/heel calibration は hidden_tail と prefix_backtest の両方で raw 系より悪化した。
- `zero_rank` と entropy は error correlation の診断材料にはなるが、直接候補置換の根拠にはならない。
- direct candidate replacement、inference port、submit は未選択。

## 実行入口

- 学習 notebook: `exp216_affine_shift_landscape_ruler_readout_train.ipynb`
- 推論 notebook: `exp216_affine_shift_landscape_ruler_readout_inference.ipynb`
