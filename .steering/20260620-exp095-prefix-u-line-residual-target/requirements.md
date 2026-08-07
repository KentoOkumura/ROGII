# 要件

## 依頼

`prefix_u_line_residual_target` を実装する。exp080 で悪化した raw U-space target の follow-up として、known prefix だけで `U_alpha = TVT + alpha * Z` の offset / slope を消した supervised target を比較できる train-side ablation を作る。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更対象は supervised target definition のみとし、exp073 feature surface、GroupKFold、LightGBM config family は固定する。
- `prefix_line(MD)` は raw train known-prefix rows の `TVT_input` と `Z` だけで fit する。
- validation tail の true `TVT` を prefix line fit、anchor、inference feature に使わない。
- 最初の比較は `alpha=1.0` と `0.5`、`lgb0` 1 model、同一 GroupKFold の `dTVT` control に限定する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `.steering/20260620-exp095-prefix-u-line-residual-target/` に要件、設計、タスクが記録されている。
- `experiments/exp095_prefix_u_line_residual_target/` に `config.yaml`、train/inference notebook、実装 `.py`、`SESSION_NOTES.md`、`result.md`、`metrics.json` がある。
- `config.yaml` に `experiment.route: ml_model`、親実験、cache parent、active targets、prefix-line fallback 条件、leakage policy が明記されている。
- train notebook は薄い `main()` 呼び出しだけでなく、設定確認、cache / prefix anchor 確認、学習実行、metrics 表示セルを持つ。
- target transform と inverse transform が `dTVT`、`prefix_u_line_alpha1p0`、`prefix_u_line_alpha0p5` で実装されている。
- prefix が短い / noisy な well は constant fallback になり、fallback 数が summary に残る。
- `validate_experiment.py`、Python compile、notebook JSON 検証が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
