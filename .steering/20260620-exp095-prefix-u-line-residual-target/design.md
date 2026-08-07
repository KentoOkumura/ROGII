# 設計

## アプローチ

exp080 の target ablation 実装をコピーし、raw U-space target ではなく known-prefix で de-trend した U-space residual target を追加する。

各 well について raw train の known-prefix rows だけを読み、`U_alpha = TVT_input + alpha * Z` を `MD` に対する robust line として fit する。評価対象 row では `prefix_line_alpha(MD)` を計算し、LightGBM は次を予測する。

- `dTVT`: `TVT - T0`
- `prefix_u_line_alpha1p0`: `(TVT + 1.0 * Z) - prefix_line_1p0(MD)`
- `prefix_u_line_alpha0p5`: `(TVT + 0.5 * Z) - prefix_line_0p5(MD)`

予測は常に TVT 空間に戻して RMSE を計算する。prefix target の inverse は `pred + prefix_line_alpha(MD) - alpha * Z`。

## 実験範囲

- 対象実験: `exp095_prefix_u_line_residual_target`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: supervised target definition only
- 固定する変数: exp073 196 features、GroupKFold by well、LightGBM lgb0 config、GPU deterministic mode、early stopping、train row set

## Prefix Line

- `alpha`: `1.0`, `0.5`
- fit input: known-prefix rows の `MD`, `Z`, `TVT_input`
- robust fit: centered `MD - anchor_md` に対する一次式を反復 MAD clipping で fit
- fallback: prefix rows が 8 未満、または MD span が 25ft 未満の場合は最後の known-prefix row の constant `U_alpha`
- evaluation row の `MD`: feature cache に `md` / `MD` があれば使い、なければ `anchor_md + md_since` で復元

## 再現性設計

- seed policy: fold split は GroupKFold で deterministic。LightGBM seed は exp073 config family を継承する。
- stochastic 処理の有無: prefix line fit は deterministic な NumPy 処理のみ。この実験内で新しい PF/Beam generation は行わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 cache に含まれる生成済み特徴のみを使用する。
- 並列処理と乱数の関係: target construction は deterministic groupby/map のみ。LightGBM は exp073 と同じ GPU deterministic mode を既定にする。
- CPU/GPU runtime と deterministic flags: `gpu_repro_guard_dp_threads8` を既定 active mode とし、`gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、固定 `num_threads=8` を使う。
- train cache / test feature regeneration の SHA 記録方針: train cache source SHA と schema SHA を summary に保存する。gzip content 比較が必要な場合は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: 各 target / model / fold の prediction SHA、保存モデル SHA、target spec summary、prefix-line fallback count を保存する。submission は本実装では作らず、推論化時に記録する。
- Kaggle package bootstrap 確認方針: push 前に `prepare_kaggle_notebooks --notebook train --run-on-push --strict` を再実行し、metadata と bootstrap 内 config が exp095 を指すことを確認する。

## リスク

- リークリスク: prefix line fit に validation tail true `TVT` を入れるとリークする。raw known-prefix rows の `TVT_input` だけを使う。
- CV/LB 不一致リスク: prefix line が hidden test の tail 変化を過度に直線化すると short tail / near rows を壊す可能性がある。distance / tail bucket と worst well を必ず見る。
- ランタイム/メモリリスク: 初回は 3 targets x lgb0 x 5 folds に限定し、exp080 の timeout を避ける。
- 再現性リスク: GPU LightGBM は bitwise 固定と決めない。採用候補になった場合は exp073 と同様に SHA と必要なら CPU control を記録する。
