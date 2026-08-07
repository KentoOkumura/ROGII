# 要件

## 依頼

`affine_shift_landscape_ruler_readout` バックログを実装し、row-level ruler readout を export する。

## 制約

- Route: `pf_beam`
- 既存の親監査 (`exp167_fft_denoised_gr_matching_audit`) の再現性と比較可能性を保つ。
- exp072 の fixed candidate cache は観測コスト readout 専用で使い、候補置換や推論には使わない。
- Kaggle Notebook 実行前提（ローカル notebook 実行はしない）。

## 受け入れ基準

- `exp216_affine_shift_landscape_ruler_readout` が作成され、`config.yaml`/`README.md`/`SESSION_NOTES.md`/`result.md`/`metrics.json` が更新済み。
- train notebook から実装済み `ruler readout` が実行できる。
- row_context に以下が保存される: `best_delta`, `second_shift_ft`, `second_delta`, `second_delta_vs_best`, `second_cost`, `margin`, `zero_shift_ft`, `zero_cost`, `zero_rank`, `secondary_mode_shift_ft`, `secondary_mode_gap`, `bimodal_flag`, `prefix_holdout_error`, `calibration_residual_scale`。
- 集約 shift curve、prefix holdout error、distance bucket 別の誤差相関が生成物として保存される。
- `task validate-exp` を通した上で、次ステップ判断用の解釈ノートに接続できる。
