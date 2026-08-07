# 設計

## アプローチ

exp146 の train-side Beam 再実行 audit を土台にする。ただし TVT+Z penalty は入れず、Beam dynamic programming が最終的に保持した上位 K 本の path と累積 cost を復元する。

各 well の exp072 pseudo-tail row に対して、raw horizontal `MD` / `GR`、prefix 最終 `TVT_input`、typewell `TVT` / `GR` だけで Beam を再実行する。最終 beam の top-K path から次を保存する。

- `*_top1_commit`
- `*_top2_commit`
- `*_topk_weighted_mean`
- `*_posterior_mean_t*`
- `*_top2_cost_gap` / `*_top2_cost_gap_per_row`
- `*_topk_entropy`
- `*_top1_top2_sep` / `*_topk_spread`
- `*_topk_oracle` と oracle rank / abs error

## 実験範囲

- 対象実験: `exp173_beam_topk_path_posterior_audit`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 診断親: `exp143`、`exp152`、`exp171`
- 変更する変数: Beam smooth radius、beam width、retained top-K posterior readout
- 固定する変数: exp072 candidate cache、train pseudo-tail row definition、scoring baseline、posterior temperature grid

## 再現性設計

- seed policy: 新しい乱数は使わない。Beam は deterministic dynamic programming。
- stochastic 処理の有無: exp173 実装内はなし。上流 exp072 PF/Beam/likPF cache は既存生成物として参照する。
- PF/Beam / likelihood-PF / seed bagging の有無: Beam 再実行あり、PF/likPF 再生成なし。
- 並列処理と乱数の関係: `num_workers=1`、global RNG 不使用。
- CPU/GPU runtime と deterministic flags: CPU notebook、GPU 不使用、LightGBM 学習なし。
- train cache / test feature regeneration の SHA 記録方針: exp072 gzip source は raw SHA と decompressed SHA を summary JSON に記録する。生成 gzip も decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: model / prediction / submission は生成しないため記録対象外。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --notebook train --strict` を使い、bootstrap 内 config と support files を正規 source から再生成する。

## リスク

- リークリスク: top-K oracle は scoring-only。候補生成や posterior temperature selection に true TVT を入れない。
- CV/LB 不一致リスク: train-side pseudo-tail audit であり、raw-test parity 未確認。positive でも inference port は別判断。
- ランタイム/メモリリスク: all-well Beam 再実行なので CPU 時間はかかるが、LightGBM / GPU booster はない。
- 再現性リスク: 上流 exp072 cache の生成履歴に依存するため deterministic submission anchor とは扱わない。
