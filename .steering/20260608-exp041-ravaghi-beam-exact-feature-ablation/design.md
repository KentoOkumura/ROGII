# 設計

## アプローチ

`exp040` の single LightGBM audit framework を継承し、`exp029` train well の途中以降を隠した疑似 test row artifact に exact Ravaghi beam feature を追加する。`exp029` には aggregate beam (`beam_pred`, spread, min/max, cost) しか保存されていないため、audit script 内で各 `(well_id, cutoff_row)` に対して train horizontal/typewell CSV を読み、Ravaghi notebook と同じ 7-config beam searchを再生成する。

生成する feature family:

- `public_beam_aggregate`: 既存 aggregate beam feature control。
- `beam_exact_paths`: `cons`, `loose`, `vcons`, `sm5`, `vloose`, `mid`, `stiff` の delta と mean/median/ref delta。
- `beam_exact_diagnostics`: path spread、range、`cons` vs `sm5` gap、final cost summary。
- `beam_exact_disagreement`: exact beam mean/ref と public beam/PF selector の disagreement。
- `pf_selector_context`: PF likelihood と selector metadata。beam exact の context としてだけ使う。

## 実験範囲

- 対象実験: `exp041_ravaghi_beam_exact_feature_ablation`
- Route: `ml_model`
- 親実験: `exp040_ravaghi_pf_ancc_pfz_feature_ablation`
- 変更する変数: exact beam path / diagnostics / disagreement feature families。
- 固定する変数: exp029 train well の途中以降を隠した疑似 test 評価条件、single LightGBM params、row-distance bucket shrink params、well-level audit surfaces、report controls。

## リスク

- リークリスク: pseudo cutoff 後の true `TVT`、error columns、exp026 OOF bridge を特徴に入れると leakage になる。実装では exact beam を target-free に再生成し、error/bridge columns を除外する。
- CV/LB 不一致リスク: exp029 train well の途中以降を隠した疑似 test 評価条件 は 見えない test well 評価の LB と一致しない可能性がある。採用条件は base single-LGBM reference を original-fold と well-hash の両方で上回ることに限定し、direct submit は別実験に分ける。
- ランタイム/メモリリスク: exact beam は well/cutoff 単位で CSV を読み直すため、full audit は Kaggle train 上で実行する。local smoke は `--max-wells` と `--max-train-rows` を使う。
