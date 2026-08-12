# 設計

## アプローチ

exp080 では U-space target が悪化したため、exp085 では supervised target を変えない。exp072 の deterministic full replay train cache を読み、PF/Beam/likelihood-PF candidate TVT を local U-space `U = candidate_tvt + Z - (T0 + Z0)` に変換する。well 内で robust polynomial を fit し、projection correction / residual / shape / disagreement を add-only features として exp073 LightGBM family に渡す。

初期実装の active variants は次の 4 つにする。

- `control_exp073_base196`
- `u_projection_correction`
- `u_disagreement`
- `u_projection_correction_plus_disagreement`

LGB OOF prediction を使う U-space features は魅力があるが、outer fold ごとに train rows 用 inner OOF と valid rows 用 base prediction を作る nested fold が必要になる。初期実装では leakage を避けるため `include_lgb_oof_features: false` に固定し、有効化された場合は runner が停止する。

## 実験範囲

- 対象実験: `exp085_u_projection_feature_ablation`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: U-space projection derived add-only feature groups
- 固定する変数: exp072 train cache、base 196 features、target `TVT - last_known_tvt`、GroupKFold by well、exp073 LightGBM config family

## 再現性設計

- seed policy: LightGBM config と GroupKFold seed は exp073/exp080 と同じ。projection feature generation は deterministic で乱数を使わない。
- stochastic 処理の有無: projection feature generation はなし。LightGBM GPU は exp073 と同じ deterministic flags / fixed thread 設定を使う。
- PF/Beam / likelihood-PF / seed bagging の有無: train 側では exp072 の生成済み deterministic cache を読む。新たな PF/Beam sampling は実行しない。
- 並列処理と乱数の関係: projection feature generation は pandas/numpy の deterministic groupby/polyfit のみ。LightGBM は config の `n_jobs` / `num_threads` に従う。
- CPU/GPU runtime と deterministic flags: default は `gpu_repro_guard_dp_threads8`。必要なら `cpu_deterministic_threads8` control を追加実行する。
- train cache / test feature regeneration の SHA 記録方針: train では exp072 cache/schema/summary SHA を manifest に記録する。inference port する場合は raw-test regenerated feature content SHA と projection feature schema を別途記録する。
- model manifest / prediction / submission SHA 記録方針: train runner が fold model SHA と OOF prediction SHA を manifest/metrics に記録する。submission は未選択なので作らない。
- Kaggle package bootstrap 確認方針: push 前に `prepare_kaggle_notebooks.py --strict` を通し、生成 metadata と bootstrap 内 config の source / GPU / internet / kernel_sources を確認する。

## リスク

- リークリスク: LGB OOF feature を雑に入れると validation fold label が train feature 生成に混ざる。初期実装では無効化して停止 guard を置く。
- CV/LB 不一致リスク: PF/Beam candidate path 由来 feature は hidden test regeneration parity の影響を受ける。train で改善しても inference port 前に test feature parity を監査する。
- ランタイム/メモリリスク: 4 variants x 3 LightGBM x 5 folds で exp080 よりは小さいが、GPU train は長い。必要なら variant を絞って再 prepare する。
- 再現性リスク: GPU LightGBM は bitwise reproducible と決めつけない。採用候補になった場合は CPU deterministic control または rerun SHA 確認を追加する。
