# 設計

## アプローチ

`exp073_gpu_reproducibility_guard_for_exp063_full_replay` を親にし、`exp039` / `exp038` 系 CV surface 上で exp073 full replay LightGBM family を再学習評価する。`exp068` の目的を引き継ぐが、対象を exp063 compact/static artifact から exp073 deterministic full replay へ差し替える。

## 実験範囲

- 対象実験: `exp076_exp039_cv_reassessment`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- 参照 branch: `exp039_ravaghi_single_lgbm_inference_submit`
- 変更する変数: 評価対象を exp073 full replay LightGBM family に変更し、exp039 CV surface で再評価する。
- 固定する変数: exp039 CV rows / folds、exp072 full replay train feature cache、exp073 stable PF/Beam seed policy、LightGBM config family。

## 再現性設計

- seed policy: PF/Beam は exp073 で確立済みの stable per-well seed policy を使う。
- stochastic 処理の有無: train は LightGBM GPU、inference は PF/Beam/likelihood-PF test feature generation。
- PF/Beam / likelihood-PF / seed bagging の有無: inference 側で raw test から full replay PF/Beam/likelihood-PF features を生成する。
- 並列処理と乱数の関係: PF/Beam は well 単位 stable seed で joblib scheduling に依存しない前提を維持する。
- CPU/GPU runtime と deterministic flags: train の主 mode は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`num_threads=8`。
- train cache / test feature regeneration の SHA 記録方針: gzip raw SHA だけでなく decompressed content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: LightGBM model SHA、OOF prediction SHA、test prediction SHA、submission SHA を summary / metrics に残す。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後、metadata の kernel_sources と support ZIP に `exp073_exp039_cv_reassessment.py` / `public_notebook_replay_audit.py` が含まれることを確認する。

## リスク

- リークリスク: exp039 `target_tvt` は scoring label としてのみ使い、feature は exp072 full replay cache から読む。
- CV/LB 不一致リスク: exp039 CV と exp073 native CV は評価面が違うため、anchor 更新根拠として混ぜない。
- ランタイム/メモリリスク: 196 features x 2 audits x 3 LightGBM config x 5 splits で重い。Kaggle GPU train を前提にする。
- 再現性リスク: PF/Beam は 2 回生成照合を行わず、確立済み seed policy と content SHA 記録を証拠にする。
