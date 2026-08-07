# 設計

## アプローチ

`exp106` の strict exp072 PF-Z parity 実装を親にする。既存 strict parity / multiseed candidate は比較基準として保持し、別 kernel `_trajectory_pf_z_seeded()` を追加する。

trajectory-aware kernel は eval-zone の raw `MD` / `Z` から `dZ/dMD` と `d2Z/dMD2` を作る。prefix の有限 `TVT_input` から prefix TVT slope、prefix `Z` slope を作る。各 row の transition では、local TVT velocity target を `beta*dZ/dMD + intercept` を基準に `d2Z/dMD2` と prefix TVT slope で補正し、高 curvature 区間では velocity / position process noise と velocity likelihood sigma を広げる。

## 実験範囲

- 対象実験: `exp142_trajectory_aware_pf_transition_prior`
- Route: `pf_beam`
- 親実験: `exp106_strict_exp072_pf_z_multiseed_scale_cache`
- 変更する変数: PF-Z transition prior の local velocity mean、velocity / position process noise、velocity likelihood sigma、trajectory variants
- 固定する変数: exp072 cache rows、typewell GR likelihood、strict parity control、scoring rows、train-side pseudo-tail evaluation

## 再現性設計

- seed policy: `stable_seed(OUTPUT_PREFIX, "trajectory_pf_z", seed_root, well, variant, seed_index)`
- stochastic 処理の有無: あり。particle initialization、process noise、resampling。
- PF/Beam / likelihood-PF / seed bagging の有無: PF-Z particle filter と seed bagging あり。
- 並列処理と乱数の関係: well-level `joblib.Parallel(... prefer="threads")` を使うが、各 well / variant の seed vector を Python 側で安定生成してから Numba kernel に渡す。
- CPU/GPU runtime と deterministic flags: CPU-only、GPU 不使用。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA と gzip decompressed SHA、raw train horizontal/typewell SHA、output SHA を summary JSON に保存する。
- model manifest / prediction / submission SHA 記録方針: 新規モデル・推論・提出なし。candidate wide と metrics の SHA を保存する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --notebook train --strict` 後に notebook JSON と packaged `.py` の compile を確認する。

## リスク

- リークリスク: eval-zone true TVT を transition prior に使うと leakage。実装では true TVT は scoring 用 `target_tvt` のみに限定する。
- CV/LB 不一致リスク: train pseudo-tail の Z-driven 改善が hidden test に再現しない可能性がある。submit する場合は raw-test-compatible inference parity を同じ exp142 内で追加する。
- ランタイム/メモリリスク: variants x seeds x particles で exp106 より重い。初期設定は 3 variants、32 seeds、600 particles に抑える。
- 再現性リスク: Numba 内 global RNG は seed 渡しで固定する。thread scheduling で乱数消費順が変わらないよう well/variant seed vector を固定する。
