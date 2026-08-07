# 設計

## アプローチ

exp095 の target-only ablation 実装を親にして、prefix U-line target を弱い linear MD/Z prior residual target に差し替える。モデル特徴量、fold、LightGBM config は exp073/exp095 と同じにし、target 定義の影響だけを見る。

## 実験範囲

- 対象実験: `exp117_linear_md_z_prior_residual_target`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- diagnostic parent: `exp113_linear_md_z_prior_global_search`
- 変更する変数: supervised target definition
- 固定する変数: train rows、196 feature cache、GroupKFold by well、LightGBM config、`lgb0` smoke

## Target 定義

- control: `dTVT = TVT - T0`
- linear residual: `target = TVT - prior`
- prior: `T0 + a * (MD - MD0) + b * (Z - Z0)`
- inverse: `pred_tvt = prior + pred_target`

最初の active targets は次に限定する。

- `dTVT`
- `linear_prior_a0p02_bm0p25`
- `linear_prior_a0p02_bm0p50`
- `linear_prior_a0p04_bm0p25`

## 再現性設計

- seed policy: exp073/exp095 と同じ fixed GroupKFold と LightGBM seeds。
- stochastic 処理の有無: LightGBM 学習のみ。target 生成は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072 cache 内の既存 feature を読むだけ。
- 並列処理と乱数の関係: GPU mode は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`num_threads=8`。
- train cache SHA: summary / manifest に source SHA と feature schema SHA を記録する。
- model manifest / prediction SHA: fold model SHA、OOF target SHA、OOF TVT SHA を記録する。
- Kaggle package bootstrap 確認: `prepare_kaggle_notebooks.py --notebook train --run-on-push --strict` を使う。

## リスク

- リークリスク: validation tail true TVT で `a,b` を fit / select すると leakage になる。係数は config 固定にする。
- CV/LB 不一致リスク: target-only improvement が hidden に転移しない可能性があるため、train-side full CV だけで inference port しない。
- ランタイム/メモリリスク: exp095 と同じ full cache / GPU LightGBM なので Kaggle train は長い。最初は `lgb0` のみ。
- 再現性リスク: GPU LightGBM の非決定性は exp073/exp095 と同じ guard 設定と SHA 記録で扱う。
