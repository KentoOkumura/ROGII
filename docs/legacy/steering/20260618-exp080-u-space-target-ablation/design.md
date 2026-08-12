# 設計

## アプローチ

exp073 の train-side LightGBM 再学習基盤を流用し、feature matrix と GroupKFold split を固定したまま target spec を差し替える。cache の既存 `target = TVT - last_known_tvt` と `last_known_tvt` から true TVT を復元し、raw train の既知 prefix 最終行から `T0` / `Z0` を well ごとに付与する。

各 target spec では LightGBM が target-space value を予測する。評価時は spec ごとの inverse transform で `pred_tvt` に戻し、RMSE は常に TVT 空間で比較する。

## 実験範囲

- 対象実験: `exp080_u_space_target_ablation`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: supervised target definition only
- 固定する変数: exp073 196 features、GroupKFold by well、LightGBM config family、GPU deterministic mode、early stopping、train row set

## Target Spec

- `dTVT`: `TVT - T0`。exp073 baseline と同義。
- `dTVT_plus_dZ`: `(TVT - T0) + (Z - Z0)`。
- `TVT_plus_Z_abs`: `TVT + Z`。
- `TVT_plus_Z_minus_T0`: `TVT + Z - T0`。
- `TVT_plus_Z_minus_T0Z0`: `TVT + Z - (T0 + Z0)`。

推論化する場合は、test raw prefix から同じ `T0` / `Z0` を復元し、選択 target の inverse transform を使う。今回の主目的は train-side target ablation であり、submission 候補化は full CV 結果確認後に判断する。

## 再現性設計

- seed policy: fold split は GroupKFold で deterministic。LightGBM seed は exp073 config family を継承する。
- stochastic 処理の有無: この実験内で新しい PF/Beam generation は行わない。train cache は exp072 deterministic cache を読む。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 cache に含まれる生成済み特徴のみを使用する。
- 並列処理と乱数の関係: target construction は deterministic groupby/map のみ。LightGBM は exp073 と同じ GPU deterministic mode を既定にする。
- CPU/GPU runtime と deterministic flags: `gpu_repro_guard_dp_threads8` を既定 active mode とし、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、固定 `num_threads=8` を使う。
- train cache / test feature regeneration の SHA 記録方針: train cache source SHA と schema SHA を summary に保存する。gzip content 比較が必要な場合は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: 各 target spec / model / fold の prediction SHA、保存モデル SHA、target spec summary を保存する。submission は本実装では作らず、推論化時に記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare_kaggle_notebooks --notebook train --run-on-push --strict` を再実行し、metadata と bootstrap 内 config が exp080 を指すことを確認する。

## リスク

- リークリスク: anchor `T0` / `Z0` を valid tail true TVT から作るとリークする。raw train の既知 prefix 最終行だけを使い、cache の `last_known_tvt` と一致することを監査する。
- CV/LB 不一致リスク: 絶対 `TVT + Z` 系 target は well 固有 offset を覚え、CV だけ改善する可能性がある。well-level / bucket metrics と target 分布を併記する。
- ランタイム/メモリリスク: 5 target x 3 LightGBM configs x 5 folds は exp073 の約 5 倍。初回は `fast=true` または target subset で smoke 可能にする。
- 再現性リスク: GPU LightGBM は bitwise 固定と決めない。採用候補になった場合は exp073 と同様に SHA と必要なら CPU control を記録する。
