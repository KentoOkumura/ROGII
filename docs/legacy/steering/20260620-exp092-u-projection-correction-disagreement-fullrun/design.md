# 設計

## アプローチ

exp085 は U-space projection derived feature の ablation を 4 variants x 3 models で実行し、Kaggle 12h 制限で timeout した。ただし logs から 59/60 fold-model metrics を回収でき、`u_projection_correction_plus_disagreement` が最有望だった。exp092 では同じ feature builder と LightGBM CV runner を再利用し、active variant を 1 つに絞って正式 pooled OOF と監査生成物を保存する。

## 実験範囲

- 対象実験: `exp092_u_projection_correction_disagreement_fullrun`
- Route: `ml_model`
- 親実験: `exp085_u_projection_feature_ablation`
- base surface parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: active feature variant を `u_projection_correction_plus_disagreement` のみに絞る。
- 固定する変数: exp072 train cache、exp073 196 base features、residual target、GroupKFold by well、LightGBM config family、GPU repro guard mode、U-projection degree/robust settings/source columns。

## 再現性設計

- seed policy: GroupKFold と LightGBM seed は exp073/exp085 family を維持する。projection feature generation は deterministic で乱数を使わない。
- stochastic 処理の有無: LightGBM 学習のみ。projection feature generation は pandas/numpy の deterministic groupby/polyfit。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam sampling はしない。exp072 deterministic cache に保存済みの PF/Beam/likelihood-PF candidate values を読むだけ。
- 並列処理と乱数の関係: feature generation は global RNG なし。LightGBM は config の fixed seed / deterministic flags / fixed thread count に従う。
- CPU/GPU runtime と deterministic flags: primary は `gpu_repro_guard_dp_threads8`。`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs=8`、`num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: train では exp072 cache/schema/summary SHA を manifest に記録する。inference port する場合は raw-test regenerated feature content SHA と projection feature schema を別途記録する。
- model manifest / prediction / submission SHA 記録方針: fold model の SHA、pooled model prediction SHA、lgb_mean prediction SHA を runner が保存する。submission はこの実験では生成しない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` で package を作り、metadata と bootstrap manifest の config / helper SHA を SESSION_NOTES に記録する。

## リスク

- リークリスク: U-projection features は candidate path / Z / MD / known-prefix anchor だけから作る。LGB OOF feature は nested fold 未実装のため無効。
- CV/LB 不一致リスク: train-side fullrun だけでは hidden test parity が未確認。inference port 前に raw-test regenerated features と train schema の一致を監査する。
- ランタイム/メモリリスク: 1 variant に絞っても 3 LightGBM x 5 folds は重い。exp085 の 4 variants timeout を踏まえ、fullrun は Kaggle GPU 上で実行する。
- 再現性リスク: GPU LightGBM は bitwise reproducible と決めつけない。採用候補にする場合は rerun または CPU deterministic control を検討する。
