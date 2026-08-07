# 設計

## アプローチ

exp073 の 196-feature full replay LightGBM surface を固定し、exp087 / exp086 で有効だった PF/Beam confidence signal を「特徴」と「学習 weight」に吸収する。PF/Beam 予測を直接 TVT として採用せず、モデルが信頼度を読める追加情報と、high-instability row の影響を弱める sample weight だけを比較する。

## 実験範囲

- 対象実験: `exp089_pf_beam_disagreement_sample_weight`
- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- 変更する変数: confidence feature group、confidence-derived sample weight policy
- 固定する変数: exp072 train cache、base 196 features、target `TVT - last_known_tvt`、GroupKFold by well、exp063/073 LightGBM config family

## Feature / Weight Design

- `confidence_core`: `pf_likpf_abs`、`pf_beam_abs`、`beam_likpf_abs`
- `confidence_context`: `beam_std_abs`、`dense_dist_abs`、`pf_vs_dense_abs`
- `confidence_score`: robust rank 平均の `pfbeam_instability_score` と派生 stable/high flags
- sample weight policy: `pfbeam_instability_score` が高い row を `0.65` 方向へ、安定 row を `1.10` 方向へ寄せ、平均 1.0 に正規化する。

## Variants

- `control_exp073_base196`: base 196 features、unit weight
- `confidence_features_core`: base + core confidence features、unit weight
- `sample_weight_unstable_downweight`: base 196 features、confidence sample weight
- `confidence_features_plus_weight`: base + core confidence features、confidence sample weight

## 再現性設計

- seed policy: fold は `GroupKFold` by well、train row subsampling が必要な場合のみ `np.random.default_rng(42)` を使う。
- stochastic 処理の有無: 新規 PF/Beam 生成はなし。LightGBM 学習のみ stochastic なので exp073 と同じ deterministic GPU flags を使う。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072 cache に保存済みの deterministic feature を読むだけで、この実験内では再生成しない。
- 並列処理と乱数の関係: LightGBM は `deterministic=true`、`force_col_wise=true`、`gpu_use_dp=true`、固定 `n_jobs/num_threads=8` を既定にする。
- CPU/GPU runtime と deterministic flags: 既定は Kaggle GPU。必要なら `cpu_deterministic_threads8` を同 config で再実行する。
- train cache / test feature regeneration の SHA 記録方針: train は exp072 cache file SHA / schema SHA を summary / manifest に記録する。inference は未選択で、選択時に raw-test regeneration parity を別途設計する。
- model manifest / prediction / submission SHA 記録方針: fold model SHA、OOF prediction SHA、lgb_mean prediction SHA を保存する。submission は未選択なので作らない。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` を通し、metadata と bootstrap manifest の config / 補助 `.py` SHA を `SESSION_NOTES.md` に記録する。

## リスク

- リークリスク: confidence feature と sample weight は target-free columns だけから作る。exp086/087 の target を使った診断結果は列選定の根拠に限定する。
- CV/LB 不一致リスク: sample weight は hidden 分布に転移しにくい可能性がある。CV 改善だけで submit せず、worst-well と bucket 悪化を確認する。
- ランタイム/メモリリスク: base 196 features に少数列を足すだけなので exp073 と同程度。
- 再現性リスク: GPU LightGBM は bitwise 固定と決めつけない。採用候補化する場合は CPU deterministic control または rerun 差分確認を行う。
