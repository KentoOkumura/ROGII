# 設計

## アプローチ

exp072 full replay train cache の `TVT_input_missing_equivalent_exp063_rows` を scoring surface として使う。raw train horizontal/typewell の GR に対して固定 filter を適用し、query well ごとに PF likelihood mean / best seed / top3 oracle diagnostic と Beam top1 を再生成する。

比較する filter は `raw`、`rolling_median_w11`、`savgol_w31_p2` の 3 つ。FFT notch と heel calibration は除外する。filter 以外の PF/Beam runtime は固定し、PF seed は well / seed index から stable SHA256 で生成して filter 間で共有する。

## 実験範囲

- 対象実験: `exp189_denoised_gr_pfbeam_generation_audit`
- Route: `pf_beam`
- 親実験: `denoised_gr_pfbeam_generation_audit` backlog
- 参照: exp072、exp099、exp167、exp170
- 変更する変数: PF/Beam observation likelihood に入れる horizontal/typewell GR filter
- 固定する変数: target well selection、eval rows、PF particles/seeds、transition noise、beam width、beam move cost、scoring metric

## 再現性設計

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_gr_filters`
- stochastic 処理の有無: PF particle propagation / resampling あり
- PF/Beam / likelihood-PF / seed bagging の有無: scoped PF と Beam rerun あり。LightGBM / seed bagging model はなし
- 並列処理と乱数の関係: 初期実装は sequential。global RNG は使わず、well / seed index から local `np.random.default_rng` を作る
- CPU/GPU runtime と deterministic flags: CPU only、Kaggle GPU disabled、internet disabled
- train cache / test feature regeneration の SHA 記録方針: exp072 input cache SHA と row candidates gzip decompressed SHA を記録する。test feature は生成しない
- model manifest / prediction / submission SHA 記録方針: model / submission は生成しない。row candidate prediction content SHA を記録する
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` で train notebook を作成し、metadata の GPU/internet と kernel source を確認する

## リスク

- リークリスク: target well selection と filter config を true error で選ぶと漏洩する。selection は eval row / md_since coverage のみを使う
- CV/LB 不一致リスク: train-side diagnostic なので Public LB anchor にはしない。inference port する場合は別途 raw-test parity が必要
- ランタイム/メモリリスク: 64 wells x 3 filters x PF 240 particles x 8 seeds と Beam を走らせる。重い場合は max_target_wells を下げる
- 再現性リスク: PF stochastic なので stable per-well seed を必須にする。filter 間の seed 共有が崩れると比較が不公平になる
