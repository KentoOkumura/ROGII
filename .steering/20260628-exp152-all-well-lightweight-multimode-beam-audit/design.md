# 設計

## アプローチ

`exp143` の PF-Z multimode 実装を再利用し、全 well に広げる代わりに診断と保存物を削る。主比較対象は従来 Beam の `exp072_beam_mean`。候補は `exp143` の scoped best だった `zacc_s010_a020_noise050` の best-likelihood seed に限定する。

## 実験範囲

- 対象実験: `exp152_all_well_lightweight_multimode_beam_audit`
- Route: `pf_beam`
- 親実験: `exp143_multimode_pfbeam_local_correlation_audit`
- 変更する変数:
  - scope を 6 well から全 train well へ広げる。
  - 各 well を tail 500 rows に制限する。
  - multimode PF-Z を 300 particles / 4 seeds / 1 variant にする。
  - local correlation と row-level mode diagnostic 保存を無効化する。
  - candidate metrics 対象を Beam 比較に必要な候補へ絞る。
- 固定する変数:
  - exp072 train feature cache。
  - raw train horizontal/typewell files。
  - train-side scoring surface。
  - ML 学習なし、提出なし。

## 再現性設計

- seed policy: `stable_sha256_seed_from_experiment_all_well_lightweight_multimode_beam_well_variant_seed_index`
- stochastic 処理の有無: あり。PF particle initialization、process noise、resampling。
- PF/Beam / likelihood-PF / seed bagging の有無: multimode PF-Z のみ。LightGBM / seed bagging はなし。
- 並列処理と乱数の関係: well 単位の thread parallel。各 well / variant / seed index に stable seed vector を事前生成する。
- CPU/GPU runtime と deterministic flags: CPU-only。GPU 不使用。Numba kernel は seed 固定だが、train-side diagnostic のため deterministic submission anchor とは扱わない。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache SHA と decompressed SHA、raw train horizontal/typewell SHA、生成物 SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: モデル、推論、提出はないため対象外。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --notebook train --strict` 後、metadata の CPU-only、internet disabled、kernel source を確認する。

## リスク

- リークリスク: 評価 zone true TVT は scoring のみに使う。候補生成は prefix `TVT_input`、GR、MD、Z、typewell のみを使う。
- CV/LB 不一致リスク: train-side pseudo-tail audit であり、本番 hidden test の直接 CV ではない。positive でもすぐ submit しない。
- ランタイム/メモリリスク: 全 well 対象のため、tail 500 rows、1 variant、4 seeds、minimal output に制限する。
- 再現性リスク: PF は stochastic。stable seed と SHA 記録で監査可能にするが、submission anchor にはしない。
