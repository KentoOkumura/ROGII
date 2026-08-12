# 設計

## アプローチ

exp070 の LightGBM reproducibility guard 実装を再利用し、train feature loader を exp072 の full replay cache に差し替える。LightGBM config family は exp063 Pixiux public replay の 3 configs を維持し、既定実行 mode は `gpu_repro_guard_dp_threads8` のみとする。加えて、inference 側の PF/Beam/likelihood-PF regeneration も reproducibility guard の対象に含める。

exp072 は train-only cache なので、inference は deterministic 化した `public_notebook_replay_audit.py` を同梱して raw test files から PF/Beam/likelihood-PF features を再生成する。PF 系乱数は well id と処理名から作る stable SHA256 seed で固定する。

## 実験範囲

- 対象実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Route: `ml_model`
- 親実験: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- cache 親実験: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: LightGBM 実行 mode (`gpu_use_dp`, deterministic flags, `num_threads`) と inference PF/Beam/likelihood-PF の seed policy。
- 固定する変数: exp063 Pixiux full replay feature surface、GroupKFold by `well`、target `TVT - last_known_tvt`、LightGBM 3-config family、PF seeds/particles。

## リスク

- リークリスク: train は exp072 の raw-train-only cache を読む。inference は train-side CV と分離し、raw test regeneration のみを行う。test PF/Beam regeneration は stable per-well seed で固定し、code-submit rerun の feature SHA / submission SHA を検証対象にする。
- CV/LB 不一致リスク: score 改善ではなく再現性監査なので、CV の微差を anchor 更新根拠にしない。prediction SHA と model SHA を重視する。
- ランタイム/メモリリスク: 196 features x 3,783,989 rows の LightGBM 15 boosters は重い。GPU package は GPU mode のみに絞り、CPU deterministic は別 package として後続判断に回す。
