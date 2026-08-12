# 設計

## アプローチ

exp211 の評価面・metrics 保存ロジックを流用しつつ、PF 本体は public replay に近い raw GR residual-scale likelihood-PF control に差し替える。改善 variant は入れず、raw public-like control だけを固定する。

## 実験範囲

- 対象実験: `exp214_public_raw_gr_residual_scale_control`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`、`exp211_affine_calibrated_gr_observation_pfbeam`、`exp213_prefix_structural_prior_pfbeam`
- 変更する変数: raw public-like likelihood-PF control の生成と scale 3/5/8/12 の保存
- 固定する変数: target well 選択、score rows、raw data、known-prefix residual scale clip 10-60、train-side scoring only

## 再現性設計

- seed policy: `stable_sha256(query_well, raw_variant, public_likpf)` を seed base とし、seed index を加える。
- stochastic 処理の有無: あり。PF particle propagation / resampling。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 500 particles x 128 seeds。Beam は raw deterministic diagnostic。
- 並列処理と乱数の関係: numba kernel 内で per-well seed base を使う。thread parallel RNG は使わない。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: output 取得時に row candidates raw gzip SHA と decompressed content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は生成しない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、metadata と config の GPU/internet/kernel source を確認する。

## リスク

- リークリスク: evaluation true TVT は scoring のみ。PF には known prefix `TVT_input` と raw GR / MD / Z だけを使う。
- CV/LB 不一致リスク: train-side diagnostic であり LB anchor ではない。raw-test regeneration と submission は別設計が必要。
- ランタイム/メモリリスク: 64 wells x 500 particles x 128 seeds。numba kernel で実装するが、Kaggle CPU runtime は要監視。
- 再現性リスク: numba RNG は seed 明示で使う。gzip SHA は raw metadata で揺れる可能性があるため decompressed SHA を主証拠にする。
